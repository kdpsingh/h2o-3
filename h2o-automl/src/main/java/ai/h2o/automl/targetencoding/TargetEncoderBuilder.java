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

      _targetEncodingMap = tec.prepareEncodingMap(_parms.train(), _parms._response_column, _parms._teFoldColumnName);

      for(Map.Entry<String, Frame> entry: _targetEncodingMap.entrySet()) {
        Scope.untrack(entry.getValue().keys());
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

  public static void printOutFrameAsTable(Frame fr) {
    printOutFrameAsTable(fr, false, fr.numRows());
  }

  public static void printOutFrameAsTable(Frame fr, boolean rollups, long limit) {
    assert limit <= Integer.MAX_VALUE;
    TwoDimTable twoDimTable = fr.toTwoDimTable(0, (int) limit, rollups);
    System.out.println(twoDimTable.toString(2, true));
  }
}
