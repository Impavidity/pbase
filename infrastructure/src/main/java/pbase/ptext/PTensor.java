package pbase.ptext;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PTensor<T> {
  long[] shape;
  List<T> value;
  public PTensor(List<T> value) {
    this.value = value;
    this.shape = Stream.of(value.size(), 1)
            .mapToLong(i->i)
            .toArray();
  }
}
