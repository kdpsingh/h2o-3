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

import java.util.Map;

import static ai.h2o.automl.targetencoding.TargetEncoderFrameHelper.convertEncodingMapToMojoFormat;


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

  public static class TargetEncoderParameters extends Model.Parameters {

    public IcedHashMap<String, Map<String, TEComponents>> _targetEncodingMap;
    
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
    
    public Boolean _withBlending = true;
    public BlendingParams _blendingParams = new BlendingParams(10, 20);

    public void addTargetEncodingMap(Map<String, Frame> encodingMap) {
      _targetEncodingMap = TargetEncoderFrameHelper.convertEncodingMapFromFrameToMap(encodingMap);
    }
    
  }

  public static class TargetEncoderOutput extends Model.Output {
    
    public transient Map<String, Map<String, int[]>> _target_encoding_map;
    public TargetEncoderParameters _teParams;
    
    public TargetEncoderOutput(TargetEncoderBuilder b) {
      super(b);
      _target_encoding_map = convertEncodingMapToMojoFormat(b._targetEncodingMap);
      _teParams = b._parms;
    }

    @Override public ModelCategory getModelCategory() {
      return ModelCategory.TargetEncoder;
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
