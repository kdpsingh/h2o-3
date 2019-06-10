package hex;

import ai.h2o.automl.targetencoding.TargetEncoderModel;
import org.junit.Ignore;
import water.H2O;
import water.Key;

import java.io.IOException;

@Ignore
public interface ModelStubs {
  
  public class TestModel extends TargetEncoderModel {
    public TestModel( Key key, TargetEncoderModel.TargetEncoderParameters p, TargetEncoderModel.TargetEncoderOutput o ) { super(key,p,o, null); }
    @Override
    public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) { throw H2O.unimpl(); }
    @Override
    protected double[] score0(double[] data, double[] preds) { throw H2O.unimpl(); }

    public static class TestParam extends TargetEncoderModel.TargetEncoderParameters {
      public String algoName() { return "A"; }
      public String fullName() { return "A"; }
      public String javaName() { return TestModel.class.getName(); }
      @Override public long progressUnits() { return 0; }
    }

    @Override
    public ModelMojoWriter getMojo() {
      return new TestMojoWriter(this);
    }
  }

  public static class TestMojoWriter extends ModelMojoWriter {
    public TestMojoWriter(TestModel model) { super(model); }
    @Override public String mojoVersion() { return "42"; }
    @Override protected void writeModelData() throws IOException { }

    @Override
    protected void writeExtraInfo() throws IOException {
      // do nothing
    }
  }
}
