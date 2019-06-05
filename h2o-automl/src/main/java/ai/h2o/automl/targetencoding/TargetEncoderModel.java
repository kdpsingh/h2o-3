package ai.h2o.automl.targetencoding;

import hex.Model;
import hex.ModelCategory;
import hex.ModelMetrics;
import hex.ModelMojoWriter;
import water.H2O;
import water.Iced;
import water.Key;
import water.fvec.Frame;

import java.util.Map;

public class TargetEncoderModel extends Model<TargetEncoderModel, TargetEncoderModel.TargetEncoderParameters, TargetEncoderModel.TargetEncoderOutput> {

  private final transient TargetEncoder _targetEncoder;
  
  public TargetEncoderModel(Key<TargetEncoderModel> selfKey, TargetEncoderParameters parms, TargetEncoderOutput output, TargetEncoder tec) {
    super(selfKey, parms, output);
    _targetEncoder = tec;
  }

  @Override
  public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    throw H2O.unimpl("No Model Metrics for TargetEncoder.");
  }

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
    
    public Boolean _withBlending = true;
    public BlendingParams _blendingParams = new BlendingParams(10, 20);
    public String[] _columnNamesToEncode;
    public String _teFoldColumnName;
  }

  public static class TargetEncoderOutput extends Model.Output {
    
    public transient Map<String, Frame> _target_encoding_map;
    public TargetEncoderParameters _teParams;
    
    public TargetEncoderOutput(TargetEncoderBuilder b) {
      super(b);
      _target_encoding_map = b._targetEncodingMap;
      _teParams = b._parms;
    }

    @Override public ModelCategory getModelCategory() {
      return ModelCategory.TargetEncoder;
    }
  }

  /**
   * Transform with noise
   */
  public Frame transform(Frame data, byte strategy, double noiseLevel, long seed){
    return _targetEncoder.applyTargetEncoding(data, _parms._response_column, this._output._target_encoding_map, strategy,
            _parms._teFoldColumnName, _parms._withBlending, noiseLevel, true, seed);
  }
  
  public Frame transform(Frame data, byte strategy, long seed){
    return _targetEncoder.applyTargetEncoding(data, _parms._response_column, this._output._target_encoding_map, strategy,
            _parms._teFoldColumnName, _parms._withBlending, true, seed);
  }
  
  @Override
  protected double[] score0(double data[], double preds[]){
    throw new UnsupportedOperationException("TargetEncoderModel doesn't support scoring. Use `transform()` instead.");
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
