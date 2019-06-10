package ai.h2o.automl.targetencoding;

import hex.ModelStubs;
import org.junit.BeforeClass;
import org.junit.Test;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import water.util.IcedHashMap;
import water.util.Log;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import static ai.h2o.automl.targetencoding.TargetEncoderFrameHelper.addKFoldColumn;
import static org.junit.Assert.*;

/**
 * This test should be moved to h2o-core module or dedicated for TE module once we move all TargetEncoding related classes there as well. 
 */

public class TargetEncoderModelTest extends TestUtil implements ModelStubs {

  @BeforeClass
  public static void stall() { stall_till_cloudsize(1); }

  String foldColumnNameForTE = "te_fold_column";

  private Map<String, Frame> getTEMapForTitanicDataset(boolean withFoldColumn) {
    Frame trainFrame = null;
    try {
      trainFrame = parse_test_file("./smalldata/gbm_test/titanic.csv");
      String responseColumnName = "survived";
      asFactor(trainFrame, responseColumnName);

      if(withFoldColumn) {
        int nfolds = 5;
        addKFoldColumn(trainFrame, foldColumnNameForTE, nfolds, 1234);
      }

      BlendingParams params = new BlendingParams(3, 1);
      String[] teColumns = {"home.dest", "embarked"};
      TargetEncoder targetEncoder = new TargetEncoder(teColumns, params);
      Map<String, Frame> testEncodingMap = targetEncoder.prepareEncodingMap(trainFrame, responseColumnName, withFoldColumn ? foldColumnNameForTE: null);
      return testEncodingMap;
    } finally {
      if(trainFrame != null) trainFrame.delete();
    }
  }

  @Test
  public void addTargetEncodingMap() {
    Scope.enter();
    try {
      TestModel.TestParam p = new TestModel.TestParam();
      Map<String, Frame> teMap = getTEMapForTitanicDataset(false);
      p.addTargetEncodingMap(teMap);

      TargetEncoderBuilder job = new TargetEncoderBuilder(p);

      TargetEncoderModel targetEncoderModel = job.trainModel().get();
      Scope.track_generic(targetEncoderModel);

      checkEncodings(targetEncoderModel._output._target_encoding_map);
      TargetEncoderFrameHelper.encodingMapCleanUp(teMap);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void getMojoFoldCase() {
    Scope.enter();
    Map<String, Frame> teMap = null;
    try {
      TestModel.TestParam p = new TestModel.TestParam();
      teMap = getTEMapForTitanicDataset(true);

      // Following iteration over encoding maps and regrouping without folds could be hidden inside `model.addTargetEncodingMap()` 
      // but we need TargetEncoder in h2o-core package  so that we can reuse functionality.
      // We need to move Target encoding to the module that `h2o-core` will be depending on. That way we can hide grouping logic into `addTargetEncoding` method.
      for (Map.Entry<String, Frame> entry : teMap.entrySet()) {
        Frame grouped = TargetEncoder.groupingIgnoringFoldColumn(foldColumnNameForTE, entry.getValue(), entry.getKey());
        entry.getValue().delete();
        teMap.put(entry.getKey(), grouped);
      }
      p.addTargetEncodingMap(teMap);

      TargetEncoderBuilder job = new TargetEncoderBuilder(p);

      TargetEncoderModel targetEncoderModel = job.trainModel().get();
      Scope.track_generic(targetEncoderModel);

      Map<String, Map<String, int[]>> targetEncodingMap = targetEncoderModel._output._target_encoding_map;

      checkEncodingsInts(targetEncodingMap);
    } finally {
      TargetEncoderFrameHelper.encodingMapCleanUp(teMap);
      Scope.exit();
    }
  }
  
  // Checking that dfork is faster
  @Test public void conversion_of_frame_into_table_doAll_vs_dfork_performance_test() {
    Map<String, Frame> encodingMap = getTEMapForTitanicDataset(false);

    for (int i = 0; i < 10; i++) { // Number of columns with encoding maps will be 2+10
      encodingMap.put(UUID.randomUUID().toString(), encodingMap.get("home.dest"));
    }
    int numberOfIterations = 20;

    //doAll
    long startTimeDoAll = System.currentTimeMillis();
    for (int i = 0; i < numberOfIterations; i++) {

      IcedHashMap<String, Map<String, TargetEncoderModel.TEComponents>> transformedEncodingMap = new IcedHashMap<>();
      for (Map.Entry<String, Frame> entry : encodingMap.entrySet()) {
        String key = entry.getKey();
        Frame encodingsForParticularColumn = entry.getValue();
        IcedHashMap<String, TargetEncoderModel.TEComponents> table = new TargetEncoderFrameHelper.FrameToTETable().doAll(encodingsForParticularColumn).getResult().table;

        transformedEncodingMap.put(key, table);
      }
    }
    long totalTimeDoAll = System.currentTimeMillis() - startTimeDoAll;
    Log.info("Total time doAll:" + totalTimeDoAll);

    //DFork
    long startTimeDFork = System.currentTimeMillis();
    for (int i = 0; i < numberOfIterations; i++) {
      Map<String, TargetEncoderFrameHelper.FrameToTETable> tasks = new HashMap<>();

      for (Map.Entry<String, Frame> entry : encodingMap.entrySet()) {
        Frame encodingsForParticularColumn = entry.getValue();
        TargetEncoderFrameHelper.FrameToTETable task = new TargetEncoderFrameHelper.FrameToTETable().dfork(encodingsForParticularColumn);

        tasks.put(entry.getKey(), task);
      }

      IcedHashMap<String, Map<String, TargetEncoderModel.TEComponents>> transformedEncodingMap = new IcedHashMap<>();

      for (Map.Entry<String, TargetEncoderFrameHelper.FrameToTETable> taskEntry : tasks.entrySet()) {
        transformedEncodingMap.put(taskEntry.getKey(), taskEntry.getValue().getResult().table);
      }
    }
    long totalTimeDFork = System.currentTimeMillis() - startTimeDFork;

    TargetEncoderFrameHelper.encodingMapCleanUp(encodingMap);
    Log.info("Total time dfork:" + totalTimeDFork);
    
    assertTrue(totalTimeDFork < totalTimeDoAll);
  }

  private void checkEncodings(Map<String, Map<String, int[]>> target_encoding_map) {
    Map<String, int[]> embarkedEncodings = target_encoding_map.get("embarked");
    Map<String, int[]> homeDestEncodings = target_encoding_map.get("home.dest");

    assertArrayEquals(embarkedEncodings.get("S"), new int[]{304, 914});
    assertArrayEquals(embarkedEncodings.get("embarked_NA"), new int[]{2, 2});
    assertEquals(homeDestEncodings.size(), 370);
  }

  private void checkEncodingsInts(Map<String, Map<String, int[]>> target_encoding_map) {
    Map<String, int[]> embarkedEncodings = target_encoding_map.get("embarked");
    Map<String, int[]> homeDestEncodings = target_encoding_map.get("home.dest");

    assertArrayEquals(embarkedEncodings.get("S"), new int[]{304, 914});
    assertArrayEquals(embarkedEncodings.get("embarked_NA"), new int[]{2, 2});
    assertEquals(homeDestEncodings.size(), 370);
  }

}
