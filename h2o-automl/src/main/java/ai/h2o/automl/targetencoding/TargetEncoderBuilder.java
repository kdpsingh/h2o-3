package ai.h2o.automl.targetencoding;

import hex.ModelBuilder;
import hex.ModelCategory;
import water.DKV;
import water.util.IcedHashMap;

import java.util.Map;

public class TargetEncoderBuilder extends ModelBuilder<TargetEncoderModel, TargetEncoderModel.TargetEncoderParameters, TargetEncoderModel.TargetEncoderOutput> {

  public transient IcedHashMap<String, Map<String, TargetEncoderModel.TEComponents>> _targetEncodingMap;
  
  public TargetEncoderBuilder(TargetEncoderModel.TargetEncoderParameters parms) {
    super(parms);
    super.init(false);
  }

  private class TargetEncoderDriver extends Driver {
    @Override
    public void computeImpl() {
      // Nothing,  but later we can perform creation of encoding map here
      _targetEncodingMap = _parms._targetEncodingMap;
      TargetEncoderModel targetEncoderModel = new TargetEncoderModel(_job._result, _parms,  new TargetEncoderModel.TargetEncoderOutput(TargetEncoderBuilder.this));
      DKV.put(targetEncoderModel);
    }
  }
  
  @Override
  protected Driver trainModelImpl() {
    // We can use Model.Parameters to configure Target Encoder
    return new TargetEncoderDriver();
  }
  
  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ ModelCategory.TargetEncoder};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }
}
