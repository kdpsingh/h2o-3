package ai.h2o.automl.targetencoding;

import hex.ModelBuilder;
import hex.ModelCategory;
import water.H2O;

public class TargetEncoderBuilder extends ModelBuilder<TargetEncoderModel, TargetEncoderModel.TargetEncoderParameters, TargetEncoderModel.TargetEncoderOutput> {

  public TargetEncoderBuilder(TargetEncoderModel.TargetEncoderParameters parms) {
    super(parms);
    super.init(false);
  }

  private class TargetEncoderDriver extends Driver {
    @Override
    public void computeImpl() {
      // Nothing,  but later we can perform creation of encoding map here
      TargetEncoderModel targetEncoderModel = new TargetEncoderModel(_job._result, _parms,  new TargetEncoderModel.TargetEncoderOutput(TargetEncoderBuilder.this));
      targetEncoderModel.delete_and_lock(_job);
      targetEncoderModel.update(_job);
      targetEncoderModel.unlock(_job);
    }
  }
  
  @Override
  protected Driver trainModelImpl() {
    // We can use Model.Parameters to configure Target Encoder
//    throw H2O.unimpl("Model is supposed to be pretrained");
    return new TargetEncoderDriver();
  }
  
//  public TargetEncoderModel getModel() {
//    return new TargetEncoderModel(_parms);
//  }
  
  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ ModelCategory.TargetEncoder};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }
}
