package ai.h2o.automl.targetencoding;

import hex.Model;
import hex.ModelMojoWriter;

import java.io.IOException;
import java.util.Map;

public class TargetEncoderMojoWriter extends ModelMojoWriter {

  public TargetEncoderMojoWriter(Model model) {
    super(model);
  }

  @Override
  public String mojoVersion() {
    return "1.00";
  }

  @Override
  protected void writeModelData() throws IOException {
    writeTargetEncodingInfo();
    writeTargetEncodingMap();
  }

  @Override
  protected void writeExtraInfo() throws IOException {
    // Do nothing
  }

  /**
   * Writes target encoding's extra info
   */
  private void writeTargetEncodingInfo() throws IOException {
    TargetEncoderModel.TargetEncoderParameters teParams = ((TargetEncoderModel) model)._output._teParams;
    writekv("with_blending", teParams._withBlending);
    if(teParams._withBlending) {
      writekv("inflection_point", teParams._blendingParams.getK());
      writekv("smoothing", teParams._blendingParams.getF());
    }
  }

  /**
   * Writes encoding map into the file line by line
   */
  private void writeTargetEncodingMap() throws IOException {
    Map<String, Map<String, int[]>> targetEncodingMap = ((TargetEncoderModel) model)._output._target_encoding_map;
    if(targetEncodingMap != null) {
      startWritingTextFile("feature_engineering/target_encoding/encoding_map.ini");
      for (Map.Entry<String, Map<String, int[]>> columnEncodingsMap : targetEncodingMap.entrySet()) {
        writeln("[" + columnEncodingsMap.getKey() + "]");
        Map<String, int[]> encodings = columnEncodingsMap.getValue();
        for (Map.Entry<String, int[]> catLevelInfo : encodings.entrySet()) {
          int[] numAndDenom = catLevelInfo.getValue();
          writelnkv(catLevelInfo.getKey(), numAndDenom[0] + " " + numAndDenom[1]);
        }
      }
      finishWritingTextFile();
    }
  }
}
