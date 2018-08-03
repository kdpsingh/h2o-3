package water.rapids.ast.prims.mungers;

import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import water.H2O;
import water.TestUtil;
import water.fvec.Frame;
import water.fvec.TestFrameBuilder;
import water.fvec.Vec;
import water.rapids.Rapids;
import water.rapids.Val;
import water.rapids.vals.ValFrame;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class AstFactorTest extends TestUtil {

  @BeforeClass
  static public void setup() { stall_till_cloudsize(1); }

  private Frame fr = null;

  @Before
  public void beforeEach() {

  }

  @Test
  public void asFactorTest() {

    fr = new TestFrameBuilder()
            .withName("testFrame")
            .withColNames("ColA")
            .withVecTypes(Vec.T_NUM)
            .withDataForCol(0, ard(0, 1))
            .build();

    assertFalse(fr.vec(0).isCategorical());

    String tree = "(as.factor testFrame)";
    Val val = Rapids.exec(tree);
    if (val instanceof ValFrame)
      fr = val.getFrame();

    assertTrue(fr.vec(0).isCategorical());
  }


  @After
  public void afterEach() {
    H2O.STORE.clear();
  }


}
