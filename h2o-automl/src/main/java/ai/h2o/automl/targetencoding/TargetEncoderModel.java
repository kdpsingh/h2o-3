package ai.h2o.automl.targetencoding;

import hex.Model;
import hex.ModelCategory;
import hex.ModelMetrics;
import hex.ModelMojoWriter;
import water.H2O;
import water.Iced;
import water.Key;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.util.IcedHashMap;

import java.util.HashMap;
import java.util.Map;


public class TargetEncoderModel extends Model<TargetEncoderModel, TargetEncoderModel.TargetEncoderParameters, TargetEncoderModel.TargetEncoderOutput> {

  public TargetEncoderModel(Key<TargetEncoderModel> selfKey, TargetEncoderParameters parms, TargetEncoderOutput output) {
    super(selfKey, parms, output);
  }

  @Override
  public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    throw H2O.unimpl("No Model Metrics for TargetEncoder.");
  }

  @Override
  protected double[] score0(double[] data, double[] preds) {
    return new double[0];
  }

  private static IcedHashMap<String, Map<String, TEComponents>> _targetEncodingMap;


  public static class TargetEncoderParameters extends Model.Parameters {
    @Override
    public String algoName() {
      return "TargetEncoder";
    }

    @Override
    public String fullName() {
      return "TargetEncoder";
    }

    @Override
    public String javaName() {
      return TargetEncoderModel.class.getName();
    }

    @Override
    public long progressUnits() {
      return 0;
    }


    public void addTargetEncodingMap(Map<String, Frame> encodingMap) {
      IcedHashMap<String, Map<String, TEComponents>> transformedEncodingMap = new IcedHashMap<>();
      Map<String, FrameToTETable> tasks = new HashMap<>();

      for (Map.Entry<String, Frame> entry : encodingMap.entrySet()) {

        Frame encodingsForParticularColumn = entry.getValue();
        FrameToTETable task = new FrameToTETable().dfork(encodingsForParticularColumn);

        tasks.put(entry.getKey(), task);
      }

      for (Map.Entry<String, FrameToTETable> taskEntry : tasks.entrySet()) {
        transformedEncodingMap.put(taskEntry.getKey(), taskEntry.getValue().getResult().table);
      }
      _targetEncodingMap = transformedEncodingMap;
    }
    
  }

  public static class TargetEncoderOutput extends Model.Output {
    public TargetEncoderOutput(TargetEncoderBuilder b) { super(b); }

    public transient Map<String, Map<String, int[]>> _target_encoding_map = convertEncodingMap(_targetEncodingMap); 


    @Override public ModelCategory getModelCategory() {
      return ModelCategory.TargetEncoder;
    }
  }


  public static Map<String, Map<String, int[]>> convertEncodingMap(IcedHashMap<String, Map<String, TEComponents>> em) {

    IcedHashMap<String, Map<String, int[]>> transformedEncodingMap = null;

      transformedEncodingMap = new IcedHashMap<>();
      for (Map.Entry<String, Map<String, TEComponents>> entry : em.entrySet()) {
        String columnName = entry.getKey();
        Map<String, TEComponents> encodingsForParticularColumn = entry.getValue();
        Map<String, int[]> encodingsForColumnMap = new HashMap<>();
        for (Map.Entry<String, TEComponents> kv : encodingsForParticularColumn.entrySet()) {
          encodingsForColumnMap.put(kv.getKey(), kv.getValue().getNumeratorAndDenominator());
        }
        transformedEncodingMap.put(columnName, encodingsForColumnMap);
      }
    return transformedEncodingMap;
  }


  static class FrameToTETable extends MRTask<FrameToTETable> {
    IcedHashMap<String, TEComponents> table = new IcedHashMap<>();

    public FrameToTETable() { }

    @Override
    public void map(Chunk[] cs) {
      Chunk categoricalChunk = cs[0];
      String[] domain = categoricalChunk.vec().domain();
      int numRowsInChunk = categoricalChunk._len;
      // Note: we don't store fold column as we need only to be able to give predictions for data which is not encoded yet. 
      // We need folds only for the case when we applying TE to the frame which we are going to train our model on. 
      // But this is done once and then we don't need them anymore.
      for (int i = 0; i < numRowsInChunk; i++) {
        int[] numeratorAndDenominator = new int[2];
        numeratorAndDenominator[0] = (int) cs[1].at8(i);
        numeratorAndDenominator[1] = (int) cs[2].at8(i);
        int factor = (int) categoricalChunk.at8(i);
        String factorName = domain[factor];
        table.put(factorName, new TEComponents(numeratorAndDenominator));
      }
    }

    @Override
    public void reduce(FrameToTETable mrt) {
      table.putAll(mrt.table);
    }
  }


  @Override
  public ModelMojoWriter getMojo() {
    return new TargetEncoderMojoWriter(this);
  }

  /**
   * Container for numerator and denominator that are being used for calculation of target encodings.
   */
  public static class TEComponents extends Iced<TEComponents> {
    private int[] _numeratorAndDenominator;
    public TEComponents(int[] numeratorAndDenominator) {
      _numeratorAndDenominator = numeratorAndDenominator;
    }

    public int[] getNumeratorAndDenominator() {
      return _numeratorAndDenominator;
    }
  }
}
