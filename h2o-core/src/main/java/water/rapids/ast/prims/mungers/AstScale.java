package water.rapids.ast.prims.mungers;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.Vec;
import water.rapids.Env;
import water.rapids.Val;
import water.rapids.vals.ValFrame;
import water.rapids.ast.AstPrimitive;
import water.rapids.ast.AstRoot;
import water.rapids.ast.params.AstNumList;
import water.util.Log;

import java.util.Arrays;

/**
 * Center and scale a frame.  Can be passed in the centers and scales (one per column in an number list), or a
 * TRUE/FALSE.
*/
public class AstScale extends AstPrimitive {
  @Override
  public String[] args() {
    return new String[]{"ary", "center", "scale"};
  }

  @Override
  public int nargs() {
    return 1 + 3;
  } // (scale x center scale)

  @Override
  public String str() {
    return "scale";
  }

  @Override
  public ValFrame apply(Env env, Env.StackHelp stk, AstRoot asts[]) {
    final Frame originalFrame = stk.track(asts[1].exec(env)).getFrame();
    final Frame numericFrame = new Frame(); // filter the frame to only numerical columns
    for (int i = 0; i < originalFrame.numCols(); i++) {
      Vec v = originalFrame.vec(i);
      if (v.get_type() == Vec.T_NUM) {
        numericFrame.add(originalFrame.name(i), v);
      }
    }

    if (numericFrame.numCols() == 0) {
      Log.info("Nothing scaled in frame '%s'. There are no numeric columns.");
      return new ValFrame(originalFrame);
    }

    final double[] means = calcMeans(env, asts[2], numericFrame, originalFrame);
    final double[] mults = calcMults(env, asts[3], numericFrame, originalFrame);
    // Update in-place.
    new InPlaceSlateTask(means, mults).doAll(numericFrame);

    // Return the original frame (with categorical and other columns), this is okay to do because the update was made in-place.
    return new ValFrame(originalFrame);
  }

  private static class InPlaceSlateTask extends MRTask<InPlaceSlateTask> {
    private final double[] _means;
    private final double[] _mults;

    InPlaceSlateTask(double[] means, double[] mults) {
      _means = means;
      _mults = mults;
    }

    @Override
    public void map(Chunk[] cs) {
      for (int i = 0; i < cs.length; i++)
        for (int row = 0; row < cs[i]._len; row++)
          cs[i].set(row, (cs[i].atd(row) - _means[i]) * _mults[i]);
    }

  }
  
  // Peel out the bias/shift/mean
  static double[] calcMeans(Env env, AstRoot meanSpec, Frame fr, Frame origFr) {
    final int ncols = fr.numCols();
    double[] means;
    if (meanSpec instanceof AstNumList) {
      means = extractNumericValues(((AstNumList) meanSpec).expand(), fr, origFr);
    } else {
      double d = meanSpec.exec(env).getNum();
      if (d == 0) means = new double[ncols]; // No change on means, so zero-filled
      else if (d == 1) means = fr.means();
      else throw new IllegalArgumentException("Only true or false allowed");
    }
    return means;
  }
  
  // Peel out the scale/stddev
  static double[] calcMults(Env env, AstRoot multSpec, Frame fr, Frame origFr) {
    double[] mults;
    if (multSpec instanceof AstNumList) {
      mults = extractNumericValues(((AstNumList) multSpec).expand(), fr, origFr);
    } else {
      Val v = multSpec.exec(env);
      if (v instanceof ValFrame) {
        mults = extractNumericValues(toArray(v.getFrame().anyVec()), fr, origFr);
      } else {
        double d = v.getNum();
        if (d == 0)
          Arrays.fill(mults = new double[fr.numCols()], 1.0); // No change on mults, so one-filled
        else if (d == 1) mults = fr.mults();
        else throw new IllegalArgumentException("Only true or false allowed");
      }
    }
    return mults;
  }
  
  private static double[] toArray(Vec v) {
    double[] res = new double[(int) v.length()];
    for (int i = 0; i < res.length; ++i)
      res[i] = v.at(i);
    return res;
  }

  private static double[] extractNumericValues(double[] vals, Frame fr, Frame origFr) {
    if (vals.length != origFr.numCols()) {
      throw new IllegalArgumentException("Values must be the same length as is the number of columns of the Frame to scale" +
              " (fill 0 for non-numeric columns).");
    }
    if (vals.length == fr.numCols())
      return vals;
    double[] numVals = new double[fr.numCols()];
    int pos = 0;
    for (int i = 0; i < origFr.numCols(); i++) {
      if (origFr.vec(i).get_type() != Vec.T_NUM)
        continue;
      numVals[pos++] = vals[i];
    }
    assert pos == numVals.length;
    return numVals;
  }

}
