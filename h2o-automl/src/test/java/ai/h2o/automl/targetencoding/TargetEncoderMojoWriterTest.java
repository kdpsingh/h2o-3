package ai.h2o.automl.targetencoding;

import hex.ModelStubs;
import org.junit.BeforeClass;
import org.junit.Test;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * This test should be moved to h2o-core module once we move all TargetEncoding related classes there as well. 
 */
public class TargetEncoderMojoWriterTest extends TestUtil implements ModelStubs {

  @BeforeClass
  public static void stall() { stall_till_cloudsize(1); }

  Frame trainFrame = parse_test_file("./smalldata/gbm_test/titanic.csv");

  private Map<String, Frame> getTEMap() {

    String responseColumnName = "survived";
    asFactor(trainFrame, responseColumnName);

    BlendingParams params = new BlendingParams(3, 1);
    String[] teColumns = {"home.dest", "embarked"};
    TargetEncoder targetEncoder = new TargetEncoder(teColumns, params);
    Map<String, Frame> testEncodingMap = targetEncoder.prepareEncodingMap(trainFrame, responseColumnName, null);
    return testEncodingMap;
  }

  @Test
  public void writeModelToZipFile() throws Exception{
    Scope.enter();
    try {
      TestModel.TestParam p = new TestModel.TestParam();
      Map<String, Frame> teMap = getTEMap();
      p.addTargetEncodingMap(teMap);

      TargetEncoderBuilder job = new TargetEncoderBuilder(p);

      TargetEncoderModel targetEncoderModel = job.trainModel().get();
      Scope.track_generic(targetEncoderModel);

      String fileName = "test_mojo_te.zip";

      try {
        FileOutputStream modelOutput = new FileOutputStream(fileName);
        targetEncoderModel.getMojo().writeTo(modelOutput);
        modelOutput.close();

        assertTrue(new File(fileName).exists());

      } finally {
        File file = new File(fileName);
        if (file.exists()) {
          file.delete();
        }
        trainFrame.delete();
        TargetEncoderFrameHelper.encodingMapCleanUp(teMap);
      }
    } finally {
      Scope.exit();
    }
  }

}
