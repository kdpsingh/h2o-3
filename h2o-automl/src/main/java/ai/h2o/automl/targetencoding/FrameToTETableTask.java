package ai.h2o.automl.targetencoding;

import water.MRTask;
import water.fvec.Chunk;
import water.util.IcedHashMap;

public class FrameToTETableTask extends MRTask<FrameToTETableTask> {
  public IcedHashMap<Integer, TEComponents> _table = new IcedHashMap<>();

  
  public FrameToTETableTask() { }

  @Override
  public void map(Chunk[] cs) {
    Chunk categoricalChunk = cs[0];
    String[] domain = categoricalChunk.vec().domain();
    int numRowsInChunk = categoricalChunk._len;
    // Note: we don't store fold column as we need only to be able to give predictions for data which is not encoded yet. 
    // We need folds only for the case when we applying TE to the frame which we are going to train our model on. 
    // But this is done once and then we don't need them anymore.
    for (int i = 0; i < numRowsInChunk; i++) {
      int[] numeratorAndDenominator = new int[2];
      numeratorAndDenominator[0] = (int) cs[1].at8(i);
      numeratorAndDenominator[1] = (int) cs[2].at8(i);
      int factor = (int) categoricalChunk.at8(i);
      _table.put(factor, new TEComponents(numeratorAndDenominator[0], numeratorAndDenominator[1]));
    }
  }

  @Override
  public void reduce(FrameToTETableTask mrt) {
    _table.putAll(mrt._table);
  }
}
