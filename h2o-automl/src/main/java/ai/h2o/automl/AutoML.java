package ai.h2o.automl;

import ai.h2o.automl.EventLogEntry.Stage;
import hex.Model;
import hex.ModelBuilder;
import hex.ScoreKeeper.StoppingMetric;
import hex.ensemble.StackedEnsembleModel;
import hex.ensemble.StackedEnsembleModel.StackedEnsembleParameters;
import hex.deeplearning.DeepLearningModel.DeepLearningParameters;
import hex.genmodel.utils.DistributionFamily;
import hex.glm.GLMModel.GLMParameters;
import hex.grid.Grid;
import hex.grid.GridSearch;
import hex.grid.HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria;
import hex.splitframe.ShuffleSplitFrame;
import hex.tree.SharedTreeModel.SharedTreeParameters;
import hex.tree.drf.DRFModel.DRFParameters;
import hex.tree.gbm.GBMModel.GBMParameters;
import hex.tree.xgboost.XGBoostModel.XGBoostParameters;
import water.*;
import water.automl.api.schemas3.AutoMLV99;
import water.exceptions.H2OIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.nbhm.NonBlockingHashMap;
import water.util.ArrayUtils;
import water.util.Countdown;
import water.util.Log;
import water.util.PrettyPrint;

import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static ai.h2o.automl.AutoMLBuildSpec.AutoMLStoppingCriteria.AUTO_STOPPING_TOLERANCE;


/**
 * H2O AutoML
 *
 * AutoML  is used for automating the machine learning workflow, which includes automatic training and
 * tuning of many models within a user-specified time-limit. Stacked Ensembles will be automatically
 * trained on collections of individual models to produce highly predictive ensemble models which, in most cases,
 * will be the top performing models in the AutoML Leaderboard.
 */
public final class AutoML extends Lockable<AutoML> implements TimedH2ORunnable {

  static class WorkAllocations extends Iced<WorkAllocations> {

    private static class Work extends Iced<Work> {
      private Algo algo;
      private int count;
      private JobType type;
      private int share;

      Work(Algo algo, int count, JobType type, int share) {
        this.algo = algo;
        this.count = count;
        this.type = type;
        this.share = share;
      }

      int consume() {
        return consume(1);
      }

      int consume(int amount) {
        int c = Math.min(this.count, amount);
        this.count -= c;
        return c * this.share;
      }

      int consumeAll() {
        return consume(Integer.MAX_VALUE);
      }
    }

    private boolean canAllocate = true;
    private Work[] allocations = new Work[0];

    WorkAllocations allocate(Algo algo, int count, JobType type, int workShare) {
      if (!canAllocate) throw new IllegalStateException("Can't allocate new work.");

      allocations = ArrayUtils.append(allocations, new Work(algo, count, type, workShare));
      return this;
    }

    void end() {
      canAllocate = false;
    }

    void remove(Algo algo) {
      List<Work> filtered = new ArrayList<>(allocations.length);
      for (Work alloc : allocations) {
        if (!algo.equals(alloc.algo)) {
          filtered.add(alloc);
        }
      }
      allocations = filtered.toArray(new Work[0]);
    }

    Work getAllocation(Algo algo, JobType workType) {
      for (Work alloc : allocations) {
        if (alloc.algo == algo && alloc.type == workType) return alloc;
      }
      return null;
    }

    private int sum(Work[] workItems) {
      int tot = 0;
      for (Work item : workItems) {
        tot += (item.count * item.share);
      }
      return tot;
    }

    int remainingWork() {
      return sum(allocations);
    }

  }

  private final static boolean verifyImmutability = true; // check that trainingFrame hasn't been messed with
  private final static SimpleDateFormat timestampFormatForKeys = new SimpleDateFormat("yyyyMMdd_HHmmss");

  /**
   * Instantiate an AutoML object and start it running.  Progress can be tracked via its job().
   *
   * @param buildSpec
   * @return
   */
  public static AutoML startAutoML(AutoMLBuildSpec buildSpec) {
    Date startTime = new Date();  // this is the one and only startTime for this run

    synchronized (AutoML.class) {
      // protect against two runs whose startTime is the same second
      if (lastStartTime != null) {
        while (lastStartTime.getYear() == startTime.getYear() &&
            lastStartTime.getMonth() == startTime.getMonth() &&
            lastStartTime.getDate() == startTime.getDate() &&
            lastStartTime.getHours() == startTime.getHours() &&
            lastStartTime.getMinutes() == startTime.getMinutes() &&
            lastStartTime.getSeconds() == startTime.getSeconds())
          startTime = new Date();
      }
      lastStartTime = startTime;
    }

    String keyString = buildSpec.build_control.project_name;
    AutoML aml = AutoML.makeAutoML(Key.<AutoML>make(keyString), startTime, buildSpec);

    DKV.put(aml);
    startAutoML(aml);
    return aml;
  }

  /**
   * Takes in an AutoML instance and starts running it. Progress can be tracked via its job().
   * @param aml
   * @return
   */
  public static void startAutoML(AutoML aml) {
    // Currently AutoML can only run one job at a time
    if (aml.job == null || !aml.job.isRunning()) {
      H2OJob j = new H2OJob(aml, aml._key, aml.runCountdown.remainingTime());
      aml.job = j._job;
      j.start(aml.workAllocations.remainingWork());
      DKV.put(aml);
    }
  }

  public static AutoML makeAutoML(Key<AutoML> key, Date startTime, AutoMLBuildSpec buildSpec) {

    AutoML autoML = new AutoML(key, startTime, buildSpec);

    if (null == autoML.trainingFrame)
      throw new H2OIllegalArgumentException("No training data has been specified, either as a path or a key.");

    return autoML;
  }

  @Override
  public Class<AutoMLV99.AutoMLKeyV3> makeSchema() {
    return AutoMLV99.AutoMLKeyV3.class;
  }

  enum JobType {
    Unknown,
    ModelBuild,
    HyperparamSearch
  }

  private AutoMLBuildSpec buildSpec;     // all parameters for doing this AutoML build
  private Frame origTrainingFrame;       // untouched original training frame

  public AutoMLBuildSpec getBuildSpec() {
    return buildSpec;
  }

  public Frame getTrainingFrame() { return trainingFrame; }
  public Frame getValidationFrame() { return validationFrame; }
  public Frame getBlendingFrame() { return blendingFrame; }
  public Frame getLeaderboardFrame() { return leaderboardFrame; }

  public Vec getResponseColumn() { return responseColumn; }
  public Vec getFoldColumn() { return foldColumn; }
  public Vec getWeightsColumn() { return weightsColumn; }

  private Frame trainingFrame;    // required training frame: can add and remove Vecs, but not mutate Vec data in place
  private Frame validationFrame;  // optional validation frame; the training_frame is split automagically if it's not specified
  private Frame blendingFrame;
  private Frame leaderboardFrame; // optional test frame used for leaderboard scoring; if not specified, leaderboard will use xval metrics

  private Vec responseColumn;
  private Vec foldColumn;
  private Vec weightsColumn;

  private Key<Grid> gridKeys[] = new Key[0];  // Grid key for the GridSearches

  private Date startTime;
  private static Date lastStartTime; // protect against two runs with the same second in the timestamp; be careful about races
  private Countdown runCountdown;
  private Job job;                  // the Job object for the build of this AutoML.

  private transient List<Job> jobs; // subjobs

  private AtomicInteger modelCount = new AtomicInteger();  // prepare for concurrency
  private Leaderboard leaderboard;
  private EventLog eventLog;

  // check that we haven't messed up the original Frame
  private Vec[] originalTrainingFrameVecs;
  private String[] originalTrainingFrameNames;
  private long[] originalTrainingFrameChecksums;

  private WorkAllocations workAllocations;

  public AutoML() {
    super(null);
  }

  public AutoML(Key<AutoML> key, Date startTime, AutoMLBuildSpec buildSpec) {
    super(key);
    this.startTime = startTime;
    this.buildSpec = buildSpec;
    this.jobs = new ArrayList<>();
    this.runCountdown = Countdown.fromSeconds(buildSpec.build_control.stopping_criteria.max_runtime_secs());

    try {
      eventLog = EventLog.make(this._key);
      eventLog().info(Stage.Workflow, "Project: " + projectName());
      eventLog().info(Stage.Workflow, "AutoML job created: " + EventLogEntry.dateTimeFormat.format(this.startTime))
              .setNamedValue("creation_epoch", this.startTime, EventLogEntry.epochFormat);

      workAllocations = planWork();

      if (null != buildSpec.input_spec.fold_column) {
        eventLog().warn(Stage.Workflow, "Custom fold column, " + buildSpec.input_spec.fold_column + ", will be used. nfolds value will be ignored.");
        buildSpec.build_control.nfolds = 0; //reset nfolds to Model default
      }

      eventLog().info(Stage.Workflow, "Build control seed: " + buildSpec.build_control.stopping_criteria.seed() +
          (buildSpec.build_control.stopping_criteria.seed() == -1 ? " (random)" : ""));

      handleDatafileParameters(buildSpec);

    if (this.buildSpec.build_control.stopping_criteria.stopping_tolerance() == AUTO_STOPPING_TOLERANCE) {
      this.buildSpec.build_control.stopping_criteria.set_default_stopping_tolerance_for_frame(this.trainingFrame);
      eventLog().info(Stage.Workflow, "Setting stopping tolerance adaptively based on the training frame: " +
              this.buildSpec.build_control.stopping_criteria.stopping_tolerance());
    } else {
      eventLog().info(Stage.Workflow, "Stopping tolerance set by the user: " + this.buildSpec.build_control.stopping_criteria.stopping_tolerance());
      double default_tolerance = AutoMLBuildSpec.AutoMLStoppingCriteria.default_stopping_tolerance_for_frame(this.trainingFrame);
      if (this.buildSpec.build_control.stopping_criteria.stopping_tolerance() < 0.7 * default_tolerance){
        eventLog().warn(Stage.Workflow, "Stopping tolerance set by the user is < 70% of the recommended default of " + default_tolerance + ", so models may take a long time to converge or may not converge at all.");
      }
    }

      String sort_metric = buildSpec.input_spec.sort_metric == null ? null : buildSpec.input_spec.sort_metric.toLowerCase();
      leaderboard = Leaderboard.getOrMake(projectName(), eventLog, this.leaderboardFrame, sort_metric);
    } catch (Exception e) {
      delete(); //cleanup potentially leaked keys
      throw e;
    }
  }


  WorkAllocations planWork() {
    if (buildSpec.build_models.exclude_algos != null && buildSpec.build_models.include_algos != null) {
      throw new  H2OIllegalArgumentException("Parameters `exclude_algos` and `include_algos` are mutually exclusive: please use only one of them if necessary.");
    }

    Set<Algo> skippedAlgos = new HashSet<>();
    if (buildSpec.build_models.exclude_algos != null) {
      skippedAlgos.addAll(Arrays.asList(buildSpec.build_models.exclude_algos));
    } else if (buildSpec.build_models.include_algos != null) {
      skippedAlgos.addAll(Arrays.asList(Algo.values()));
      skippedAlgos.removeAll(Arrays.asList(buildSpec.build_models.include_algos));
    }

    for (Algo algo : Algo.values()) {
      if (!skippedAlgos.contains(algo) && !algo.enabled()) {
        eventLog.warn(Stage.ModelTraining, "AutoML: "+algo.name()+" is not available; skipping it.");
        skippedAlgos.add(algo);
      }
    }

    WorkAllocations workAllocations = new WorkAllocations();
    workAllocations.allocate(Algo.DeepLearning, 1, JobType.ModelBuild, 10)
            .allocate(Algo.DeepLearning, 3, JobType.HyperparamSearch, 20)
            .allocate(Algo.DRF, 2, JobType.ModelBuild, 10)
            .allocate(Algo.GBM, 5, JobType.ModelBuild, 10)
            .allocate(Algo.GBM, 1, JobType.HyperparamSearch, 60)
            .allocate(Algo.GLM, 1, JobType.HyperparamSearch, 20)
            .allocate(Algo.XGBoost, 3, JobType.ModelBuild, 10)
            .allocate(Algo.XGBoost, 1, JobType.HyperparamSearch, 100)
            .allocate(Algo.StackedEnsemble, 2, JobType.ModelBuild, 15)
            .end();

    for (Algo skippedAlgo : skippedAlgos) {
      eventLog().info(Stage.ModelTraining, "Disabling Algo: "+skippedAlgo+" as requested by the user.");
      workAllocations.remove(skippedAlgo);
    }

    return workAllocations;
  }

  @Override
  public void run() {
    runCountdown.start();
    eventLog().info(Stage.Workflow, "AutoML build started: " + EventLogEntry.dateTimeFormat.format(runCountdown.start_time()))
            .setNamedValue("start_epoch", runCountdown.start_time(), EventLogEntry.epochFormat);
    learn();
    stop();
  }

  @Override
  public void stop() {
    if (null == jobs) return; // already stopped
    for (Job j : jobs) j.stop();
    for (Job j : jobs) j.get(); // Hold until they all completely stop.
    jobs = null;

    runCountdown.stop();
    eventLog().info(Stage.Workflow, "AutoML build stopped: " + EventLogEntry.dateTimeFormat.format(runCountdown.stop_time()))
            .setNamedValue("stop_epoch", runCountdown.stop_time(), EventLogEntry.epochFormat);
    eventLog().info(Stage.Workflow, "AutoML build done: built " + modelCount + " models");
    eventLog().info(Stage.Workflow, "AutoML duration: "+ PrettyPrint.msecs(runCountdown.duration(), true))
            .setNamedValue("duration_secs", Math.round(runCountdown.duration() / 1000.));

    Log.info(eventLog().toString("Event Log for AutoML Run " + this._key + ":"));
    for (EventLogEntry event : eventLog()._events)
      Log.info(event);

    if (0 < this.leaderboard().getModelKeys().length) {
      Log.info(leaderboard().toTwoDimTable("Leaderboard for project " + projectName(), true).toString());
    }

    possiblyVerifyImmutability();
    if (!buildSpec.build_control.keep_cross_validation_predictions) {
      cleanUpModelsCVPreds();
    }
  }

  private void learn() {
    defaultXGBoosts(false);
    defaultSearchGLM(null);
    defaultRandomForest();
//    defaultXGBoosts(true);
    defaultGBMs();
    defaultDeepLearning();
    defaultExtremelyRandomTrees();
    defaultSearchXGBoost(null, false);
//    defaultSearchXGBoost(null, true);
    defaultSearchGBM(null);
    defaultSearchDL();
    defaultStackedEnsembles();
  }

  /**
   * Holds until AutoML's job is completed, if a job exists.
   */
  public void get() {
    if (job != null) job.get();
  }

  public Job job() {
    if (null == this.job) return null;
    return DKV.getGet(this.job._key);
  }

  public Model leader() { return (leaderboard() == null ? null : leaderboard.getLeader()); }

  public Leaderboard leaderboard() {
    if (leaderboard != null) leaderboard = leaderboard._key.get();
    return leaderboard;
  }

  public EventLog eventLog() {
    if (eventLog != null) eventLog = eventLog._key.get();
    return eventLog;
  }

  public String projectName() {
    return buildSpec == null ? null : buildSpec.project();
  }

  public long timeRemainingMs() {
    return runCountdown.remainingTime();
  }

  public int remainingModels() {
    if (buildSpec.build_control.stopping_criteria.max_models() == 0)
      return Integer.MAX_VALUE;
    return buildSpec.build_control.stopping_criteria.max_models() - modelCount.get();
  }

  @Override
  public boolean keepRunning() {
    return !runCountdown.timedOut() && remainingModels() > 0;
  }

  private boolean isCVEnabled() {
    return this.buildSpec.build_control.nfolds != 0 || this.buildSpec.input_spec.fold_column != null;
  }


  //*****************  Data Preparation Section  *****************//

  private void optionallySplitTrainingDataset() {
    // If no cross-validation and validation or leaderboard frame are missing,
    // then we need to create one out of the original training set.
    if (!isCVEnabled()) {
      double[] splitRatios = null;
      if (null == this.validationFrame && null == this.leaderboardFrame) {
        splitRatios = new double[]{ 0.8, 0.1, 0.1 };
        eventLog().info(Stage.DataImport,
            "Since cross-validation is disabled, and none of validation frame and leaderboard frame were provided, " +
                "automatically split the training data into training, validation and leaderboard frames in the ratio 80/10/10");
      } else if (null == this.validationFrame) {
        splitRatios = new double[]{ 0.9, 0.1, 0 };
        eventLog().info(Stage.DataImport,
            "Since cross-validation is disabled, and no validation frame was provided, " +
                "automatically split the training data into training and validation frames in the ratio 90/10");
      } else if (null == this.leaderboardFrame) {
        splitRatios = new double[]{ 0.9, 0, 0.1 };
        eventLog().info(Stage.DataImport,
            "Since cross-validation is disabled, and no leaderboard frame was provided, " +
                "automatically split the training data into training and leaderboard frames in the ratio 90/10");
      }
      if (splitRatios != null) {
        Key[] keys = new Key[] {
            Key.make("automl_training_"+origTrainingFrame._key),
            Key.make("automl_validation_"+origTrainingFrame._key),
            Key.make("automl_leaderboard_"+origTrainingFrame._key),
        };
        Frame[] splits = ShuffleSplitFrame.shuffleSplitFrame(
            origTrainingFrame,
            keys,
            splitRatios,
            buildSpec.build_control.stopping_criteria.seed()
        );
        this.trainingFrame = splits[0];

        if (this.validationFrame == null && splits[1].numRows() > 0) {
          this.validationFrame = splits[1];
        } else {
          splits[1].delete();
        }

        if (this.leaderboardFrame == null && splits[2].numRows() > 0) {
          this.leaderboardFrame = splits[2];
        } else {
          splits[2].delete();
        }
      }
    }
  }
  private void handleDatafileParameters(AutoMLBuildSpec buildSpec) {
    this.origTrainingFrame = DKV.getGet(buildSpec.input_spec.training_frame);
    this.validationFrame = DKV.getGet(buildSpec.input_spec.validation_frame);
    this.blendingFrame = DKV.getGet(buildSpec.input_spec.blending_frame);
    this.leaderboardFrame = DKV.getGet(buildSpec.input_spec.leaderboard_frame);

    Map<String, Frame> compatible_frames = new LinkedHashMap(){{
      put("training", origTrainingFrame);
      put("validation", validationFrame);
      put("blending", blendingFrame);
      put("leaderboard", leaderboardFrame);
    }};
    for (Map.Entry<String, Frame> entry : compatible_frames.entrySet()) {
      Frame frame = entry.getValue();
      if (frame != null && frame.find(buildSpec.input_spec.response_column) == -1) {
        throw new H2OIllegalArgumentException("Response column '"+buildSpec.input_spec.response_column+"' is not in the "+entry.getKey()+" frame.");
      }
    }

    if (buildSpec.input_spec.fold_column != null && this.origTrainingFrame.find(buildSpec.input_spec.fold_column) == -1) {
      throw new H2OIllegalArgumentException("Fold column '"+buildSpec.input_spec.fold_column+"' is not in the training frame.");
    }
    if (buildSpec.input_spec.weights_column != null && this.origTrainingFrame.find(buildSpec.input_spec.weights_column) == -1) {
      throw new H2OIllegalArgumentException("Weights column '"+buildSpec.input_spec.weights_column+"' is not in the training frame.");
    }

    optionallySplitTrainingDataset();

    if (null == this.trainingFrame) {
      // when nfolds>0, let trainingFrame be the original frame
      // but cloning to keep an internal ref just in case the original ref gets deleted from client side
      // (can occur in some corner cases with Python GC for example if frame get's out of scope during an AutoML rerun)
      this.trainingFrame = new Frame(origTrainingFrame);
      this.trainingFrame._key = Key.make("automl_training_" + origTrainingFrame._key);
      DKV.put(this.trainingFrame);
    }

    this.responseColumn = trainingFrame.vec(buildSpec.input_spec.response_column);
    this.foldColumn = trainingFrame.vec(buildSpec.input_spec.fold_column);
    this.weightsColumn = trainingFrame.vec(buildSpec.input_spec.weights_column);

    this.eventLog().info(Stage.DataImport,
        "training frame: "+this.trainingFrame.toString().replace("\n", " ")+" checksum: "+this.trainingFrame.checksum());
    if (null != this.validationFrame) {
      this.eventLog().info(Stage.DataImport,
          "validation frame: "+this.validationFrame.toString().replace("\n", " ")+" checksum: "+this.validationFrame.checksum());
    } else {
      this.eventLog().info(Stage.DataImport, "validation frame: NULL");
    }
    if (null != this.leaderboardFrame) {
      this.eventLog().info(Stage.DataImport,
          "leaderboard frame: "+this.leaderboardFrame.toString().replace("\n", " ")+" checksum: "+this.leaderboardFrame.checksum());
    } else {
      this.eventLog().info(Stage.DataImport, "leaderboard frame: NULL");
    }

    this.eventLog().info(Stage.DataImport, "response column: "+buildSpec.input_spec.response_column);
    this.eventLog().info(Stage.DataImport, "fold column: "+this.foldColumn);
    this.eventLog().info(Stage.DataImport, "weights column: "+this.weightsColumn);

    if (verifyImmutability) {
      // check that we haven't messed up the original Frame
      originalTrainingFrameVecs = origTrainingFrame.vecs().clone();
      originalTrainingFrameNames = origTrainingFrame.names().clone();
      originalTrainingFrameChecksums = new long[originalTrainingFrameVecs.length];

      for (int i = 0; i < originalTrainingFrameVecs.length; i++)
        originalTrainingFrameChecksums[i] = originalTrainingFrameVecs[i].checksum();
    }
    DKV.put(this);
  }


  //*****************  Jobs Build / Configure / Run / Poll section (model agnostic, kind of...) *****************//


  private void pollAndUpdateProgress(Stage stage, String name, WorkAllocations.Work work, Job parentJob, Job subJob) {
    pollAndUpdateProgress(stage, name, work, parentJob, subJob, false);
  }

  private void pollAndUpdateProgress(Stage stage, String name, WorkAllocations.Work work, Job parentJob, Job subJob, boolean ignoreTimeout) {
    if (null == subJob) {
      if (null != parentJob) {
        parentJob.update(work.consume(), "SKIPPED: " + name);
        Log.info("AutoML skipping " + name);
      }
      return;
    }
    eventLog().debug(stage, name + " started");
    jobs.add(subJob);

    long lastWorkedSoFar = 0;
    long lastTotalGridModelsBuilt = 0;

    while (subJob.isRunning()) {
      if (null != parentJob) {
        if (parentJob.stop_requested()) {
          eventLog().debug(stage, "AutoML job cancelled; skipping " + name);
          subJob.stop();
        }
        if (!ignoreTimeout && runCountdown.timedOut()) {
          eventLog().debug(stage, "AutoML: out of time; skipping " + name);
          subJob.stop();
        }
      }
      long workedSoFar = Math.round(subJob.progress() * work.share);

      if (null != parentJob) {
        parentJob.update(Math.round(workedSoFar - lastWorkedSoFar), name);
      }

      if (JobType.HyperparamSearch == work.type) {
        Grid<?> grid = (Grid)subJob._result.get();
        int totalGridModelsBuilt = grid.getModelCount();
        if (totalGridModelsBuilt > lastTotalGridModelsBuilt) {
          eventLog().debug(stage, "Built: " + totalGridModelsBuilt + " models for search: " + name);
          this.addModels(grid.getModelKeys());
          lastTotalGridModelsBuilt = totalGridModelsBuilt;
        }
      }

      try {
        Thread.sleep(1000);
      }
      catch (InterruptedException e) {
        // keep going
      }
      lastWorkedSoFar = workedSoFar;
    }

    // pick up any stragglers:
    if (JobType.HyperparamSearch == work.type) {
      if (subJob.isCrashed()) {
        eventLog().warn(stage, name + " failed: " + subJob.ex().toString());
      } else if (subJob.get() == null) {
        eventLog().info(stage, name + " cancelled");
      } else {
        Grid<?> grid = (Grid) subJob.get();
        int totalGridModelsBuilt = grid.getModelCount();
        if (totalGridModelsBuilt > lastTotalGridModelsBuilt) {
          eventLog().debug(stage, "Built: " + totalGridModelsBuilt + " models for search: " + name);
          this.addModels(grid.getModelKeys());
        }
        eventLog().debug(stage, name + " complete");
      }
    } else if (JobType.ModelBuild == work.type) {
      if (subJob.isCrashed()) {
        eventLog().warn(stage, name + " failed: " + subJob.ex().toString());
      } else if (subJob.get() == null) {
        eventLog().info(stage, name + " cancelled");
      } else {
        eventLog().debug(stage, name + " complete");
        this.addModel((Model) subJob.get());
      }
    }

    // add remaining work
    if (null != parentJob) {
      parentJob.update(work.share - lastWorkedSoFar);
    }
    work.consume();
    jobs.remove(subJob);
  }

  // These are per (possibly concurrent) AutoML run.
  // All created keys for a run use the unique AutoML run timestamp, so we can't have name collisions.
  private int individualModelsTrained = 0;
  private NonBlockingHashMap<String, Integer> algoInstanceCounters = new NonBlockingHashMap<>();
  private NonBlockingHashMap<String, Integer> gridInstanceCounters = new NonBlockingHashMap<>();

  private int nextInstanceCounter(String algoName, NonBlockingHashMap<String, Integer> instanceCounters) {
    synchronized (instanceCounters) {
      int instanceNum = 1;
      if (instanceCounters.containsKey(algoName))
        instanceNum = instanceCounters.get(algoName) + 1;
      instanceCounters.put(algoName, instanceNum);
      return instanceNum;
    }
  }

  private Key<Model> modelKey(String algoName) {
    return modelKey(algoName, true);
  }

  private Key<Model> modelKey(String algoName, boolean with_counter) {
    String counterStr = with_counter ? "_" + nextInstanceCounter(algoName, algoInstanceCounters) : "";
    return Key.make(algoName + counterStr + "_AutoML_" + timestampFormatForKeys.format(this.startTime));
  }

  Job<Model> trainModel(Key<Model> key, WorkAllocations.Work work, Model.Parameters parms) {
    return trainModel(key, work, parms, false);
  }

  /**
   * @param key (optional) model key
   * @param work  the allocated work item: used to check various executions limits and to distribute remaining work (time + models)
   * @param parms the model builder params
   * @param ignoreLimits (defaults to false) whether or not to ignore the max_models/max_runtime constraints
   * @return a started training model
   */
  Job<Model> trainModel(Key<Model> key, WorkAllocations.Work work, Model.Parameters parms, boolean ignoreLimits) {
    if (exceededSearchLimits(work, key == null ? null : key.toString(), ignoreLimits)) return null;

    Algo algo = work.algo;
    String algoName = ModelBuilder.algoName(algo.urlName());

    if (null == key) key = modelKey(algoName);

    Job<Model> job = new Job<>(key, ModelBuilder.javaName(algo.urlName()), algoName);
    ModelBuilder builder = ModelBuilder.make(algo.urlName(), job, key);
    Model.Parameters defaults = builder._parms;
    builder._parms = parms;

    setCommonModelBuilderParams(builder._parms);

    if (ignoreLimits)
      builder._parms._max_runtime_secs = 0;
    else if (builder._parms._max_runtime_secs == 0)
      builder._parms._max_runtime_secs = timeRemainingMs() / 1e3;
    else
      builder._parms._max_runtime_secs = Math.min(builder._parms._max_runtime_secs, timeRemainingMs() / 1e3);

    setStoppingCriteria(parms, defaults);

    // If we have set a seed for the search and not for the individual model params
    // then use a sequence starting with the same seed given for the model build.
    // Don't use the same exact seed so that, e.g., if we build two GBMs they don't
    // do the same row and column sampling.
    if (builder._parms._seed == defaults._seed && buildSpec.build_control.stopping_criteria.seed() != -1)
      builder._parms._seed = buildSpec.build_control.stopping_criteria.seed() + individualModelsTrained++;

    builder.init(false);          // validate parameters

    // TODO: handle error_count and messages

    Log.debug("Training model: " + algoName + ", time remaining (ms): " + timeRemainingMs());
    try {
      return builder.trainModelOnH2ONode();
    } catch (H2OIllegalArgumentException exception) {
      eventLog().warn(Stage.ModelTraining, "Skipping training of model "+key+" due to exception: "+exception);
      return null;
    }
  }

  private Key<Grid> gridKey(String algoName) {
    return Key.make(algoName + "_grid_" + nextInstanceCounter(algoName, gridInstanceCounters) + "_AutoML_" + timestampFormatForKeys.format(this.startTime));
  }

  private void addGridKey(Key<Grid> gridKey) {
    gridKeys = Arrays.copyOf(gridKeys, gridKeys.length + 1);
    gridKeys[gridKeys.length - 1] = gridKey;
  }

  /**
   * Do a random hyperparameter search.  Caller must eventually do a <i>get()</i>
   * on the returned Job to ensure that it's complete.
   * @param gridKey optional grid key
   * @param work  the allocated work item: used to check various executions limits and to distribute remaining work (time + models)
   * @param baseParms ModelBuilder parameter values that are common across all models in the search
   * @param searchParms hyperparameter search space
   * @return the started hyperparameter search job
   */
  Job<Grid> hyperparameterSearch(
      Key<Grid> gridKey, WorkAllocations.Work work, Model.Parameters baseParms, Map<String, Object[]> searchParms
  ) {
    if (exceededSearchLimits(work)) return null;

    Algo algo = work.algo;
    setCommonModelBuilderParams(baseParms);

    RandomDiscreteValueSearchCriteria searchCriteria = (RandomDiscreteValueSearchCriteria) buildSpec.build_control.stopping_criteria.getSearchCriteria().clone();
    float remainingWorkRatio = (float) work.share / workAllocations.remainingWork();
    double maxAssignedTime = remainingWorkRatio * timeRemainingMs() / 1e3;
    int maxAssignedModels = (int) Math.ceil(remainingWorkRatio * remainingModels());

    if (searchCriteria.max_runtime_secs() == 0)
      searchCriteria.set_max_runtime_secs(maxAssignedTime);
    else
      searchCriteria.set_max_runtime_secs(Math.min(searchCriteria.max_runtime_secs(), maxAssignedTime));

    if (searchCriteria.max_models() == 0)
      searchCriteria.set_max_models(maxAssignedModels);
    else
      searchCriteria.set_max_models(Math.min(searchCriteria.max_models(), maxAssignedModels));

    eventLog().info(Stage.ModelTraining, "AutoML: starting " + algo + " hyperparameter search");

    Model.Parameters defaults;
    try {
      defaults = baseParms.getClass().newInstance();
    } catch (Exception e) {
      eventLog().warn(Stage.ModelTraining, "Internal error doing hyperparameter search");
      throw new H2OIllegalArgumentException("Hyperparameter search can't create a new instance of Model.Parameters subclass: " + baseParms.getClass());
    }

    setStoppingCriteria(baseParms, defaults);

    // NOTE:
    // RandomDiscrete Hyperparameter Search matches the logic used in #trainModel():
    // If we have set a seed for the search and not for the individual model params
    // then use a sequence starting with the same seed given for the model build.
    // Don't use the same exact seed so that, e.g., if we build two GBMs they don't
    // do the same row and column sampling.
    if (null == gridKey) gridKey = gridKey(algo.name());
    addGridKey(gridKey);
    Log.debug("Hyperparameter search: " + algo.name() + ", time remaining (ms): " + timeRemainingMs());
    return GridSearch.startGridSearch(
        gridKey,
        baseParms,
        searchParms,
        new GridSearch.SimpleParametersBuilderFactory<>(),
        searchCriteria
    );
  }

  Job<StackedEnsembleModel> stack(String modelName, Key<Model>[] modelKeyArrays, boolean use_cache) {
    WorkAllocations.Work work = workAllocations.getAllocation(Algo.StackedEnsemble, JobType.ModelBuild);
    if (work == null) return null;

    // Set up Stacked Ensemble
    StackedEnsembleParameters stackedEnsembleParameters = new StackedEnsembleParameters();
    stackedEnsembleParameters._base_models = modelKeyArrays;
    stackedEnsembleParameters._valid = (getValidationFrame() == null ? null : getValidationFrame()._key);
    stackedEnsembleParameters._blending = (getBlendingFrame() == null ? null : getBlendingFrame()._key);
    stackedEnsembleParameters._keep_levelone_frame = true; //TODO Why is this true? Can be optionally turned off
    stackedEnsembleParameters._keep_base_model_predictions = use_cache; //avoids recomputing some base predictions for each SE
    // Add cross-validation args
    stackedEnsembleParameters._metalearner_fold_column = buildSpec.input_spec.fold_column;
    stackedEnsembleParameters._metalearner_nfolds = buildSpec.build_control.nfolds;

    stackedEnsembleParameters.initMetalearnerParams();
    stackedEnsembleParameters._metalearner_parameters._keep_cross_validation_models = buildSpec.build_control.keep_cross_validation_models;
    stackedEnsembleParameters._metalearner_parameters._keep_cross_validation_predictions = buildSpec.build_control.keep_cross_validation_predictions;

    Key modelKey = modelKey(modelName, false);
    return trainModel(modelKey, work, stackedEnsembleParameters, true);
  }

  private void setCommonModelBuilderParams(Model.Parameters params) {
    params._train = trainingFrame._key;
    if (null != validationFrame)
      params._valid = validationFrame._key;
    params._response_column = buildSpec.input_spec.response_column;
    params._ignored_columns = buildSpec.input_spec.ignored_columns;
    params._seed = buildSpec.build_control.stopping_criteria.seed();
    params._max_runtime_secs = buildSpec.build_control.stopping_criteria.max_runtime_secs_per_model();

    // currently required, for the base_models, for stacking:
    if (! (params instanceof StackedEnsembleParameters)) {
      params._keep_cross_validation_predictions = getBlendingFrame() == null ? true : buildSpec.build_control.keep_cross_validation_predictions;

      // TODO: StackedEnsemble doesn't support weights yet in score0
      params._fold_column = buildSpec.input_spec.fold_column;
      params._weights_column = buildSpec.input_spec.weights_column;

      if (buildSpec.input_spec.fold_column == null) {
        params._nfolds = buildSpec.build_control.nfolds;
        if (buildSpec.build_control.nfolds > 1) {
          // TODO: below allow the user to specify this (vs Modulo)
          params._fold_assignment = Model.Parameters.FoldAssignmentScheme.Modulo;
        }
      }
      if (buildSpec.build_control.balance_classes) {
        params._balance_classes = buildSpec.build_control.balance_classes;
        params._class_sampling_factors = buildSpec.build_control.class_sampling_factors;
        params._max_after_balance_size = buildSpec.build_control.max_after_balance_size;
      }
      //TODO: add a check that gives an error when class_sampling_factors, max_after_balance_size is set and balance_classes = false
    }

    params._keep_cross_validation_models = buildSpec.build_control.keep_cross_validation_models;
    params._keep_cross_validation_fold_assignment = buildSpec.build_control.nfolds != 0 && buildSpec.build_control.keep_cross_validation_fold_assignment;
    params._export_checkpoints_dir = buildSpec.build_control.export_checkpoints_dir;
  }

  private void setStoppingCriteria(Model.Parameters parms, Model.Parameters defaults) {
    // If the caller hasn't set ModelBuilder stopping criteria, set it from our global criteria.

    //FIXME: Do we really need to compare with defaults before setting the buildSpec value instead?
    // This can create subtle bugs: e.g. if dev wanted to enforce a stopping criteria for a specific algo/model,
    // he wouldn't be able to enforce the default value, that would always be overridden by buildSpec.
    // We should instead provide hooks and ensure that properties are always set in the following order:
    //  1. defaults, 2. user defined, 3. internal logic/algo specific based on the previous state (esp. handling of AUTO properties).
    if (parms._stopping_metric == defaults._stopping_metric)
      parms._stopping_metric = buildSpec.build_control.stopping_criteria.stopping_metric();

    if (parms._stopping_metric == StoppingMetric.AUTO) {
      String sort_metric = getSortMetric();
      parms._stopping_metric = sort_metric == null ? StoppingMetric.AUTO
                              : sort_metric.equals("auc") ? StoppingMetric.logloss
                              : metricValueOf(sort_metric);
    }

    if (parms._stopping_rounds == defaults._stopping_rounds)
      parms._stopping_rounds = buildSpec.build_control.stopping_criteria.stopping_rounds();

    if (parms._stopping_tolerance == defaults._stopping_tolerance)
      parms._stopping_tolerance = buildSpec.build_control.stopping_criteria.stopping_tolerance();
  }

  private boolean exceededSearchLimits(WorkAllocations.Work work) {
    return exceededSearchLimits(work, null, false);
  }

  private boolean exceededSearchLimits(WorkAllocations.Work work, String algo_desc, boolean ignoreLimits) {
    String fullName = algo_desc == null ? work.algo.toString() : work.algo+" ("+algo_desc+")";
    if (!ignoreLimits && runCountdown.timedOut()) {
      eventLog().debug(Stage.ModelTraining, "AutoML: out of time; skipping "+fullName+" in "+work.type);
      return true;
    }

    if (!ignoreLimits && remainingModels() <= 0) {
      eventLog().debug(Stage.ModelTraining, "AutoML: hit the max_models limit; skipping "+fullName+" in "+work.type);
      return true;
    }
    return false;
  }


  //*****************  Default Models Section *****************//


  void defaultXGBoosts(boolean emulateLightGBM) {
    Algo algo = Algo.XGBoost;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.ModelBuild);
    if (work == null) return;

    XGBoostParameters xgBoostParameters = new XGBoostParameters();
    setCommonModelBuilderParams(xgBoostParameters);

    Job xgBoostJob;
    Key<Model> key;

    if (emulateLightGBM) {
      xgBoostParameters._tree_method = XGBoostParameters.TreeMethod.hist;
      xgBoostParameters._grow_policy = XGBoostParameters.GrowPolicy.lossguide;
    }

    // setDistribution: no way to identify gaussian, poisson, laplace? using descriptive statistics?
    xgBoostParameters._distribution = getResponseColumn().isBinary() && !(getResponseColumn().isNumeric()) ? DistributionFamily.bernoulli
                    : getResponseColumn().isCategorical() ? DistributionFamily.multinomial
                    : DistributionFamily.AUTO;

    xgBoostParameters._score_tree_interval = 5;
    xgBoostParameters._stopping_rounds = 5;
//    xgBoostParameters._stopping_tolerance = Math.min(1e-2, RandomDiscreteValueSearchCriteria.default_stopping_tolerance_for_frame(this.trainingFrame));

    xgBoostParameters._ntrees = 10000;
    xgBoostParameters._learn_rate = 0.05;
//    xgBoostParameters._min_split_improvement = 0.01f;

    //XGB 1 (medium depth)
    xgBoostParameters._max_depth = 10;
    xgBoostParameters._min_rows = 5;
    xgBoostParameters._sample_rate = 0.6;
    xgBoostParameters._col_sample_rate = 0.8;
    xgBoostParameters._col_sample_rate_per_tree = 0.8;

    if (emulateLightGBM) {
      xgBoostParameters._max_leaves = 1 << xgBoostParameters._max_depth;
      xgBoostParameters._max_depth = xgBoostParameters._max_depth * 2;
//      xgBoostParameters._min_data_in_leaf = (float) xgBoostParameters._min_rows;
      xgBoostParameters._min_sum_hessian_in_leaf = (float) xgBoostParameters._min_rows;
    }

    key = modelKey(algo.name());
    xgBoostJob = trainModel(key, work, xgBoostParameters);
    pollAndUpdateProgress(Stage.ModelTraining,  key.toString(), work, this.job(), xgBoostJob);

    //XGB 2 (deep)
    xgBoostParameters._max_depth = 20;
    xgBoostParameters._min_rows = 10;
    xgBoostParameters._sample_rate = 0.6;
    xgBoostParameters._col_sample_rate = 0.8;
    xgBoostParameters._col_sample_rate_per_tree = 0.8;

    if (emulateLightGBM) {
      xgBoostParameters._max_leaves = 1 << xgBoostParameters._max_depth;
      xgBoostParameters._max_depth = xgBoostParameters._max_depth * 2;
//      xgBoostParameters._min_data_in_leaf = (float) xgBoostParameters._min_rows;
      xgBoostParameters._min_sum_hessian_in_leaf = (float) xgBoostParameters._min_rows;
    }

    key = modelKey(algo.name());
    xgBoostJob = trainModel(key, work, xgBoostParameters);
    pollAndUpdateProgress(Stage.ModelTraining,  key.toString(), work, this.job(), xgBoostJob);

    //XGB 3 (shallow)
    xgBoostParameters._max_depth = 5;
    xgBoostParameters._min_rows = 3;
    xgBoostParameters._sample_rate = 0.8;
    xgBoostParameters._col_sample_rate = 0.8;
    xgBoostParameters._col_sample_rate_per_tree = 0.8;

    if (emulateLightGBM) {
      xgBoostParameters._max_leaves = 1 << xgBoostParameters._max_depth;
      xgBoostParameters._max_depth = xgBoostParameters._max_depth * 2;
//      xgBoostParameters._min_data_in_leaf = (float) xgBoostParameters._min_rows;
      xgBoostParameters._min_sum_hessian_in_leaf = (float) xgBoostParameters._min_rows;
    }

    key = modelKey(algo.name());
    xgBoostJob = trainModel(key, work, xgBoostParameters);
    pollAndUpdateProgress(Stage.ModelTraining,  key.toString(), work, this.job(), xgBoostJob);
  }


  void defaultSearchXGBoost(Key<Grid> gridKey, boolean emulateLightGBM) {
    Algo algo = Algo.XGBoost;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.HyperparamSearch);
    if (work == null) return;

    XGBoostParameters xgBoostParameters = new XGBoostParameters();
    setCommonModelBuilderParams(xgBoostParameters);


    if (emulateLightGBM) {
       xgBoostParameters._tree_method = XGBoostParameters.TreeMethod.hist;
      xgBoostParameters._grow_policy = XGBoostParameters.GrowPolicy.lossguide;
    }

    xgBoostParameters._distribution = getResponseColumn().isBinary() && !(getResponseColumn().isNumeric()) ? DistributionFamily.bernoulli
            : getResponseColumn().isCategorical() ? DistributionFamily.multinomial
            : DistributionFamily.AUTO;

    xgBoostParameters._score_tree_interval = 5;
    xgBoostParameters._stopping_rounds = 5;
//    xgBoostParameters._stopping_tolerance = Math.min(1e-2, RandomDiscreteValueSearchCriteria.default_stopping_tolerance_for_frame(this.trainingFrame));

    xgBoostParameters._ntrees = 10000;
    xgBoostParameters._learn_rate = 0.05;
//    xgBoostParameters._min_split_improvement = 0.01f; //DAI default

    Map<String, Object[]> searchParams = new HashMap<>();
//    searchParams.put("_ntrees", new Integer[]{100, 1000, 10000}); // = _n_estimators

    if (emulateLightGBM) {
      searchParams.put("_max_leaves", new Integer[]{1<<5, 1<<10, 1<<15, 1<<20});
      searchParams.put("_max_depth", new Integer[]{10, 20, 50});
      searchParams.put("_min_sum_hessian_in_leaf", new Double[]{0.01, 0.1, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0});
    } else {
      searchParams.put("_max_depth", new Integer[]{5, 10, 15, 20});
      searchParams.put("_min_rows", new Double[]{0.01, 0.1, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0});  // = _min_child_weight
    }

    searchParams.put("_sample_rate", new Double[]{0.6, 0.8, 1.0}); // = _subsample
    searchParams.put("_col_sample_rate" , new Double[]{ 0.6, 0.8, 1.0}); // = _colsample_bylevel"
    searchParams.put("_col_sample_rate_per_tree", new Double[]{ 0.7, 0.8, 0.9, 1.0}); // = _colsample_bytree: start higher to always use at least about 40% of columns
//    searchParams.put("_learn_rate", new Double[]{0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0}); // = _eta
//    searchParams.put("_min_split_improvement", new Float[]{0.01f, 0.05f, 0.1f, 0.5f, 1f, 5f, 10f, 50f}); // = _gamma
//    searchParams.put("_tree_method", new XGBoostParameters.TreeMethod[]{XGBoostParameters.TreeMethod.auto});
    searchParams.put("_booster", new XGBoostParameters.Booster[]{ //gblinear crashes currently
            XGBoostParameters.Booster.gbtree, //default, let's use it more often
            XGBoostParameters.Booster.gbtree,
            XGBoostParameters.Booster.dart
    });

    searchParams.put("_reg_lambda", new Float[]{0.001f, 0.01f, 0.1f, 1f, 10f, 100f});
    searchParams.put("_reg_alpha", new Float[]{0.001f, 0.01f, 0.1f, 0.5f, 1f});

    Job<Grid> xgBoostSearchJob = hyperparameterSearch(gridKey, work, xgBoostParameters, searchParams);
    pollAndUpdateProgress(Stage.ModelTraining, algo.name()+" hyperparameter search", work, this.job(), xgBoostSearchJob);
  }


  void defaultRandomForest() {
    Algo algo = Algo.DRF;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.ModelBuild);
    if (work == null) return;

    DRFParameters drfParameters = new DRFParameters();
    setCommonModelBuilderParams(drfParameters);
    drfParameters._stopping_tolerance = this.buildSpec.build_control.stopping_criteria.stopping_tolerance();

    Job randomForestJob = trainModel(null, work, drfParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "Default Random Forest build", work, this.job(), randomForestJob);
  }


  void defaultExtremelyRandomTrees() {
    Algo algo = Algo.DRF;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.ModelBuild);
    if (work == null) return;

    DRFParameters drfParameters = new DRFParameters();
    setCommonModelBuilderParams(drfParameters);
    drfParameters._histogram_type = SharedTreeParameters.HistogramType.Random;
    drfParameters._stopping_tolerance = this.buildSpec.build_control.stopping_criteria.stopping_tolerance();

    Job randomForestJob = trainModel(modelKey("XRT"), work, drfParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "Extremely Randomized Trees (XRT) Random Forest build", work, this.job(), randomForestJob);
  }


  /**
   * Build Arno's magical 5 default GBMs.
   */
  void defaultGBMs() {
    Algo algo = Algo.GBM;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.ModelBuild);
    if (work == null) return;

    Job gbmJob;

    GBMParameters gbmParameters = new GBMParameters();
    setCommonModelBuilderParams(gbmParameters);
    gbmParameters._score_tree_interval = 5;
    gbmParameters._histogram_type = SharedTreeParameters.HistogramType.AUTO;

    gbmParameters._ntrees = 1000;
    gbmParameters._sample_rate = 0.8;
    gbmParameters._col_sample_rate = 0.8;
    gbmParameters._col_sample_rate_per_tree = 0.8;

    // Default 1:
    gbmParameters._max_depth = 6;
    gbmParameters._min_rows = 1;

    gbmJob = trainModel(null, work, gbmParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "GBM 1", work, this.job(), gbmJob);

    // Default 2:
    gbmParameters._max_depth = 7;
    gbmParameters._min_rows = 10;

    gbmJob = trainModel(null, work, gbmParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "GBM 2", work, this.job(), gbmJob);

    // Default 3:
    gbmParameters._max_depth = 8;
    gbmParameters._min_rows = 10;

    gbmJob = trainModel(null, work, gbmParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "GBM 3", work, this.job(), gbmJob);

    // Default 4:
    gbmParameters._max_depth = 10;
    gbmParameters._min_rows = 10;

    gbmJob = trainModel(null, work, gbmParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "GBM 4", work, this.job(), gbmJob);

    // Default 5:
    gbmParameters._max_depth = 15;
    gbmParameters._min_rows = 100;

    gbmJob = trainModel(null, work, gbmParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "GBM 5", work, this.job(), gbmJob);
  }


  void defaultDeepLearning() {
    Algo algo = Algo.DeepLearning;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.ModelBuild);
    if (work == null) return;

    DeepLearningParameters deepLearningParameters = new DeepLearningParameters();
    setCommonModelBuilderParams(deepLearningParameters);
    deepLearningParameters._stopping_tolerance = this.buildSpec.build_control.stopping_criteria.stopping_tolerance();
    deepLearningParameters._hidden = new int[]{ 10, 10, 10 };

    Job deepLearningJob = trainModel(null, work, deepLearningParameters);
    pollAndUpdateProgress(Stage.ModelTraining, "Default Deep Learning build", work, this.job(), deepLearningJob);
  }


  void defaultSearchGLM(Key<Grid> gridKey) {
    Algo algo = Algo.GLM;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.HyperparamSearch);
    if (work == null) return;

    GLMParameters glmParameters = new GLMParameters();
    setCommonModelBuilderParams(glmParameters);
    glmParameters._lambda_search = true;
    glmParameters._family =
            getResponseColumn().isBinary() && !(getResponseColumn().isNumeric()) ? GLMParameters.Family.binomial
            : getResponseColumn().isCategorical() ? GLMParameters.Family.multinomial
            : GLMParameters.Family.gaussian;  // TODO: other continuous distributions!

    Map<String, Object[]> searchParams = new HashMap<>();
    glmParameters._alpha = new double[] {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};  // Note: standard GLM parameter is an array; don't use searchParams!
    // NOTE: removed MissingValuesHandling.Skip for now because it's crashing.  See https://0xdata.atlassian.net/browse/PUBDEV-4974
    searchParams.put("_missing_values_handling", new DeepLearningParameters.MissingValuesHandling[] {DeepLearningParameters.MissingValuesHandling.MeanImputation /* , DeepLearningModel.DeepLearningParameters.MissingValuesHandling.Skip */});

    Job<Grid> glmJob = hyperparameterSearch(gridKey, work, glmParameters, searchParams);
    pollAndUpdateProgress(Stage.ModelTraining, "GLM hyperparameter search", work, this.job(), glmJob);
  }

  void defaultSearchGBM(Key<Grid> gridKey) {
    Algo algo = Algo.GBM;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.HyperparamSearch);
    if (work == null) return;

    GBMParameters gbmParameters = new GBMParameters();
    setCommonModelBuilderParams(gbmParameters);
    gbmParameters._score_tree_interval = 5;
    gbmParameters._histogram_type = SharedTreeParameters.HistogramType.AUTO;

    Map<String, Object[]> searchParams = new HashMap<>();
    searchParams.put("_ntrees", new Integer[]{10000});
    searchParams.put("_max_depth", new Integer[]{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
    searchParams.put("_min_rows", new Integer[]{1, 5, 10, 15, 30, 100});
    searchParams.put("_learn_rate", new Double[]{0.001, 0.005, 0.008, 0.01, 0.05, 0.08, 0.1, 0.5, 0.8});
    searchParams.put("_sample_rate", new Double[]{0.50, 0.60, 0.70, 0.80, 0.90, 1.00});
    searchParams.put("_col_sample_rate", new Double[]{ 0.4, 0.7, 1.0});
    searchParams.put("_col_sample_rate_per_tree", new Double[]{ 0.4, 0.7, 1.0});
    searchParams.put("_min_split_improvement", new Double[]{1e-4, 1e-5});

    Job<Grid> gbmJob = hyperparameterSearch(gridKey, work, gbmParameters, searchParams);
    pollAndUpdateProgress(Stage.ModelTraining, "GBM hyperparameter search", work, this.job(), gbmJob);
  }

  void defaultSearchDL() {
    Key<Grid> dlGridKey = gridKey(Algo.DeepLearning.name());
    defaultSearchDL1(dlGridKey);
    defaultSearchDL2(dlGridKey);
    defaultSearchDL3(dlGridKey);
  }

  void defaultSearchDL1(Key<Grid> gridKey) {
    Algo algo = Algo.DeepLearning;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.HyperparamSearch);
    if (work == null) return;

    DeepLearningParameters dlParameters = new DeepLearningParameters();
    setCommonModelBuilderParams(dlParameters);
    dlParameters._epochs = 10000; // early stopping takes care of epochs - no need to tune!
    dlParameters._adaptive_rate = true;
    dlParameters._activation = DeepLearningParameters.Activation.RectifierWithDropout;

    Map<String, Object[]> searchParams = new HashMap<>();
    // common:
    searchParams.put("_rho", new Double[] { 0.9, 0.95, 0.99 });
    searchParams.put("_epsilon", new Double[] { 1e-6, 1e-7, 1e-8, 1e-9 });
    searchParams.put("_input_dropout_ratio", new Double[] { 0.0, 0.05, 0.1, 0.15, 0.2 });
    // unique:
    searchParams.put("_hidden", new Integer[][] { {50}, {200}, {500} });
    searchParams.put("_hidden_dropout_ratios", new Double[][] { { 0.0 }, { 0.1 }, { 0.2 }, { 0.3 }, { 0.4 }, { 0.5 } });

    Job<Grid>dlJob = hyperparameterSearch(gridKey, work, dlParameters, searchParams);
    pollAndUpdateProgress(Stage.ModelTraining, "DeepLearning hyperparameter search 1", work, this.job(), dlJob);
  }

  void defaultSearchDL2(Key<Grid> gridKey) {
    Algo algo = Algo.DeepLearning;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.HyperparamSearch);
    if (work == null) return;

    DeepLearningParameters dlParameters = new DeepLearningParameters();
    setCommonModelBuilderParams(dlParameters);
    dlParameters._epochs = 10000; // early stopping takes care of epochs - no need to tune!
    dlParameters._adaptive_rate = true;
    dlParameters._activation = DeepLearningParameters.Activation.RectifierWithDropout;

    Map<String, Object[]> searchParams = new HashMap<>();
    // common:
    searchParams.put("_rho", new Double[] { 0.9, 0.95, 0.99 });
    searchParams.put("_epsilon", new Double[] { 1e-6, 1e-7, 1e-8, 1e-9 });
    searchParams.put("_input_dropout_ratio", new Double[] { 0.0, 0.05, 0.1, 0.15, 0.2 });
    // unique:
    searchParams.put("_hidden", new Integer[][] { {50, 50}, {200, 200}, {500, 500} });
    searchParams.put("_hidden_dropout_ratios", new Double[][] { { 0.0, 0.0 }, { 0.1, 0.1 }, { 0.2, 0.2 }, { 0.3, 0.3 }, { 0.4, 0.4 }, { 0.5, 0.5 } });

    Job<Grid>dlJob = hyperparameterSearch(gridKey, work, dlParameters, searchParams);
    pollAndUpdateProgress(Stage.ModelTraining, "DeepLearning hyperparameter search 2", work, this.job(), dlJob);
  }

  void defaultSearchDL3(Key<Grid> gridKey) {
    Algo algo = Algo.DeepLearning;
    WorkAllocations.Work work = workAllocations.getAllocation(algo, JobType.HyperparamSearch);
    if (work == null) return;

    DeepLearningParameters dlParameters = new DeepLearningParameters();
    setCommonModelBuilderParams(dlParameters);
    dlParameters._epochs = 10000; // early stopping takes care of epochs - no need to tune!
    dlParameters._adaptive_rate = true;
    dlParameters._activation = DeepLearningParameters.Activation.RectifierWithDropout;

    Map<String, Object[]> searchParams = new HashMap<>();
    // common:
    searchParams.put("_rho", new Double[] { 0.9, 0.95, 0.99 });
    searchParams.put("_epsilon", new Double[] { 1e-6, 1e-7, 1e-8, 1e-9 });
    searchParams.put("_input_dropout_ratio", new Double[] { 0.0, 0.05, 0.1, 0.15, 0.2 });
    // unique:
    searchParams.put("_hidden", new Integer[][] { {50, 50, 50}, {200, 200, 200}, {500, 500, 500} });
    searchParams.put("_hidden_dropout_ratios", new Double[][] { { 0.0, 0.0, 0.0 }, { 0.1, 0.1, 0.1 }, { 0.2, 0.2, 0.2 }, { 0.3, 0.3, 0.3 }, { 0.4, 0.4, 0.4 }, { 0.5, 0.5, 0.5 } });

    Job<Grid>dlJob = hyperparameterSearch(gridKey, work, dlParameters, searchParams);
    pollAndUpdateProgress(Stage.ModelTraining, "DeepLearning hyperparameter search 3", work, this.job(), dlJob);
  }

  void defaultStackedEnsembles() {
    Model[] allModels = leaderboard().getModels();

    WorkAllocations.Work seWork = workAllocations.getAllocation(Algo.StackedEnsemble, JobType.ModelBuild);
    if (seWork == null) {
      this.job.update(0, "StackedEnsemble builds skipped");
      eventLog().info(Stage.ModelTraining, "StackedEnsemble builds skipped due to the exclude_algos option.");
    } else if (allModels.length == 0) {
      this.job.update(seWork.consumeAll(), "No models built; StackedEnsemble builds skipped");
      eventLog().info(Stage.ModelTraining, "No models were built, due to timeouts or the exclude_algos option. StackedEnsemble builds skipped.");
    } else if (allModels.length == 1) {
      this.job.update(seWork.consumeAll(), "One model built; StackedEnsemble builds skipped");
      eventLog().info(Stage.ModelTraining, "StackedEnsemble builds skipped since there is only one model built");
    } else if (!isCVEnabled() && getBlendingFrame() == null) {
      this.job.update(seWork.consumeAll(), "Cross-validation disabled by the user and no blending frame provided; StackedEnsemble build skipped");
      eventLog().info(Stage.ModelTraining,"Cross-validation disabled by the user and no blending frame provided; StackedEnsemble build skipped");
    } else {
      // Also stack models from other AutoML runs, by using the Leaderboard! (but don't stack stacks)
      int nonEnsembleCount = 0;
      for (Model aModel : allModels)
        if (!(aModel instanceof StackedEnsembleModel))
          nonEnsembleCount++;

      Key<Model>[] notEnsembles = new Key[nonEnsembleCount];
      int notEnsembleIndex = 0;
      for (Model aModel : allModels)
        if (!(aModel instanceof StackedEnsembleModel))
          notEnsembles[notEnsembleIndex++] = aModel._key;

      // Set aside List<Model> for best models per model type. Meaning best GLM, GBM, DRF, XRT, and DL (5 models).
      // This will give another ensemble that is smaller than the original which takes all models into consideration.
      List<Model> bestModelsOfEachType = new ArrayList<>();
      Set<String> typesOfGatheredModels = new HashSet<>();

      for (Model aModel : allModels) {
        String type = getModelType(aModel);
        if (aModel instanceof StackedEnsembleModel || typesOfGatheredModels.contains(type)) continue;
        typesOfGatheredModels.add(type);
        bestModelsOfEachType.add(aModel);
      }

      Key<Model>[] bestModelKeys = new Key[bestModelsOfEachType.size()];
      for (int i = 0; i < bestModelsOfEachType.size(); i++)
        bestModelKeys[i] = bestModelsOfEachType.get(i)._key;

      Job<StackedEnsembleModel> bestEnsembleJob = stack("StackedEnsemble_BestOfFamily", bestModelKeys, true);
      pollAndUpdateProgress(Stage.ModelTraining, "StackedEnsemble build using top model from each algorithm type", seWork, this.job(), bestEnsembleJob, true);

      Job<StackedEnsembleModel> ensembleJob = stack("StackedEnsemble_AllModels", notEnsembles, false);
      pollAndUpdateProgress(Stage.ModelTraining, "StackedEnsemble build using all AutoML models", seWork, this.job(), ensembleJob, true);
    }
  }

  //*****************  Clean Up + other utility functions *****************//

  /**
   * Delete the AutoML-related objects, but leave the grids and models that it built.
   */
  @Override
  protected Futures remove_impl(Futures fs) {
    Key<Job> jobKey = job == null ? null : job._key;

    if (gridKeys != null)
      for (Key<Grid> gridKey : gridKeys) gridKey.remove(fs);

    // If the Frame was made here (e.g. buildspec contained a path, then it will be deleted
    if (buildSpec.input_spec.training_frame == null && origTrainingFrame != null) {
      origTrainingFrame.delete(jobKey, fs);
    }
    if (buildSpec.input_spec.validation_frame == null && validationFrame != null) {
      validationFrame.delete(jobKey, fs);
    }
    if (buildSpec.input_spec.leaderboard_frame == null && leaderboardFrame != null) {
      leaderboardFrame.delete(jobKey, fs);
    }

    if (trainingFrame != null && origTrainingFrame != null)
      Frame.deleteTempFrameAndItsNonSharedVecs(trainingFrame, origTrainingFrame);
    if (leaderboard() != null) leaderboard().remove(fs);
    if (eventLog() != null) eventLog().remove(fs);

    return super.remove_impl(fs);
  }

  // If we have multiple AutoML engines running on the same project they will be updating the Leaderboard concurrently,
  // so always use leaderboard() instead of the raw field, to get it from the DKV.
  // Also, the leaderboard will reject duplicate models, so use the difference in Leaderboard length here.
  private void addModels(final Key<Model>[] newModels) {
    int before = leaderboard().getModelCount();
    leaderboard().addModels(newModels);
    int after = leaderboard().getModelCount();
    modelCount.addAndGet(after - before);
  }

  private void addModel(final Model newModel) {
    int before = leaderboard().getModelCount();
    leaderboard().addModel(newModel);
    int after = leaderboard().getModelCount();
    modelCount.addAndGet(after - before);
  }

  private String getSortMetric() {
    //ensures that the sort metric is always updated according to the defaults set by leaderboard
    Leaderboard leaderboard = leaderboard();
    return leaderboard == null ? null : leaderboard.sort_metric;
  }

  private static StoppingMetric metricValueOf(String name) {
    if (name == null) return StoppingMetric.AUTO;
    switch (name) {
      case "mean_residual_deviance": return StoppingMetric.deviance;
      default:
        String[] attempts = { name, name.toUpperCase(), name.toLowerCase() };
        for (String attempt : attempts) {
          try {
            return StoppingMetric.valueOf(attempt);
          } catch (IllegalArgumentException ignored) { }
        }
        return StoppingMetric.AUTO;
    }
  }

  private boolean possiblyVerifyImmutability() {
    boolean warning = false;

    if (verifyImmutability) {
      // check that we haven't messed up the original Frame
      eventLog().debug(Stage.Workflow, "Verifying training frame immutability. . .");

      Vec[] vecsRightNow = origTrainingFrame.vecs();
      String[] namesRightNow = origTrainingFrame.names();

      if (originalTrainingFrameVecs.length != vecsRightNow.length) {
        Log.warn("Training frame vec count has changed from: " +
                originalTrainingFrameVecs.length + " to: " + vecsRightNow.length);
        warning = true;
      }
      if (originalTrainingFrameNames.length != namesRightNow.length) {
        Log.warn("Training frame vec count has changed from: " +
                originalTrainingFrameNames.length + " to: " + namesRightNow.length);
        warning = true;
      }

      for (int i = 0; i < originalTrainingFrameVecs.length; i++) {
        if (!originalTrainingFrameVecs[i].equals(vecsRightNow[i])) {
          Log.warn("Training frame vec number " + i + " has changed keys.  Was: " +
                  originalTrainingFrameVecs[i] + " , now: " + vecsRightNow[i]);
          warning = true;
        }
        if (!originalTrainingFrameNames[i].equals(namesRightNow[i])) {
          Log.warn("Training frame vec number " + i + " has changed names.  Was: " +
                  originalTrainingFrameNames[i] + " , now: " + namesRightNow[i]);
          warning = true;
        }
        if (originalTrainingFrameChecksums[i] != vecsRightNow[i].checksum()) {
          Log.warn("Training frame vec number " + i + " has changed checksum.  Was: " +
                  originalTrainingFrameChecksums[i] + " , now: " + vecsRightNow[i].checksum());
          warning = true;
        }
      }

      if (warning)
        eventLog().warn(Stage.Workflow, "Training frame was mutated!  This indicates a bug in the AutoML software.");
      else
        eventLog().debug(Stage.Workflow, "Training frame was not mutated (as expected).");

    } else {
      eventLog().debug(Stage.Workflow, "Not verifying training frame immutability. . .  This is turned off for efficiency.");
    }

    return warning;
  }

  private String getModelType(Model m) {
    return m._key.toString().startsWith("XRT_") ? "XRT" : m._parms.algoName();
  }

  private void cleanUpModelsCVPreds() {
    Log.info("Cleaning up all CV Predictions for AutoML");
    for (Model model : leaderboard().getModels()) {
        model.deleteCrossValidationPreds();
    }
  }
}
