package ai.h2o.automl.targetencoding;

import hex.ModelBuilder;
import hex.ModelCategory;
import water.DKV;
import water.Scope;
import water.fvec.Frame;
import water.util.TwoDimTable;

import java.util.Map;

public class TargetEncoderBuilder extends ModelBuilder<TargetEncoderModel, TargetEncoderModel.TargetEncoderParameters, TargetEncoderModel.TargetEncoderOutput> {

  public transient Map<String, Frame> _targetEncodingMap;
  
  public TargetEncoderBuilder(TargetEncoderModel.TargetEncoderParameters parms) {
    super(parms);
    super.init(false);
  }

  private class TargetEncoderDriver extends Driver {
    @Override
    public void computeImpl() {
      
      TargetEncoder tec = new TargetEncoder(_parms._columnNamesToEncode, _parms._blendingParams);

      Scope.untrack(train().keys());
      
      _targetEncodingMap = tec.prepareEncodingMap(train(), _parms._response_column, _parms._teFoldColumnName);

      for(Map.Entry<String, Frame> entry: _targetEncodingMap.entrySet()) {
        Frame frameWithEncodingMap = entry.getValue();
        Scope.untrack(frameWithEncodingMap.keys());
      }
      
      TargetEncoderModel targetEncoderModel = new TargetEncoderModel(_job._result, _parms,  new TargetEncoderModel.TargetEncoderOutput(TargetEncoderBuilder.this), tec);
      DKV.put(targetEncoderModel);
    }
  }
  
  @Override
  protected Driver trainModelImpl() {
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
