package pbase.ptext;

import java.util.ArrayList;
import java.util.List;

public class PaddedTensor {
  List paddedTensor;
  List lengths;
  List<Long> shape;

  public PaddedTensor(List paddedTensor, List lengths) {
    this.paddedTensor = paddedTensor;
    this.lengths = lengths;
    this.shape = getShape(this.paddedTensor);
  }

  public List<Long> getShape(Object tensor) {
    List<Long> shape = new ArrayList<>();
    System.out.println(tensor.getClass());
    while (tensor.getClass() == ArrayList.class) {
      shape.add(Long.valueOf(((List)tensor).size()));
      tensor = ((List)tensor).get(0);
    }
    return shape;
  }


}
