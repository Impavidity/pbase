package pbase.ptext;


import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;


public class NestedField extends Field {
  Field nestingField;

  public NestedField(
          Field nestingField,
          Object initToken,
          Object eosToken,
          Function<String, List> tokenize,
          Object padToken,
          int fixLength
  ) {
    super(true,
            initToken,
            eosToken,
            false,
            tokenize,
            nestingField.sequential ? nestingField.padToken : padToken,
            fixLength);
    this.nestingField = nestingField;
  }

  public Object preprocess(Object example) {
    List<Object> results = new ArrayList<>();
    for (Object x: (List)super.preprocess(example)) {
      results.add(this.nestingField.preprocess(x));
    }
    return results;
  }

  public PaddedTensor pad(List batch) {
    if (!this.nestingField.sequential)
      return super.pad(batch);
    if (this.nestingField.fixLength < 0) {
      int maxLen = 0;
      for (Object example: batch)
        for (Object x: (List)example) {
          int len = size(x);
          if (len > maxLen) maxLen = len;
      }
      this.nestingField.fixLength = maxLen +
              ((this.nestingField.initToken == null) ? 0 : 1) +
              ((this.nestingField.eosToken == null) ? 0 : 1);
    }
    if (this.initToken != null) {
      List<Object> initExamples = new ArrayList<>();
      initExamples.add(this.initToken);
      List<List<Object>> initExamplesBatch = new ArrayList<>();
      initExamplesBatch.add(initExamples);
      this.initToken = (this.nestingField.pad(initExamplesBatch)).paddedTensor.get(0);
    }
    if (this.eosToken != null) {
      List<Object> eosExamples = new ArrayList<>();
      eosExamples.add(this.eosToken);
      List<List<Object>> eosExamplesBatch = new ArrayList<>();
      eosExamplesBatch.add(eosExamples);
      this.eosToken = (this.nestingField.pad(eosExamplesBatch)).paddedTensor.get(0);
    }


    List<Object> nestedPaddedList = new ArrayList<>();
    List<Object> lengths = new ArrayList<>();
    for (Object example: batch) {
      PaddedTensor paddedTensor = this.nestingField.pad((List) example);
      nestedPaddedList.add(paddedTensor.paddedTensor);
      lengths.add(paddedTensor.lengths);
    }
    List<Object> padTokenList = new ArrayList<>();
    for (int i=0; i<this.nestingField.fixLength; i++)
      padTokenList.add(this.nestingField.padToken);
    this.padToken = padTokenList;
    PaddedTensor paddedTensor = super.pad(nestedPaddedList);
    int maxLen = 0;
    int maxLenIdx = -1;
    for (int i=0; i<paddedTensor.lengths.size(); i++)
      if (maxLen < (int)paddedTensor.lengths.get(i)) {
        maxLen = (int) paddedTensor.lengths.get(i);
        maxLenIdx = i;
      }
    if (this.fixLength != -1) maxLen = this.fixLength;
    Object padLenToken = ((List)lengths.get(maxLenIdx)).get(0);
    Object zeroPadLenToken = fillZeros(padLenToken);
    for (int i=0; i<lengths.size(); i++)
      while (((List)lengths.get(i)).size() < maxLen)
        ((List)lengths.get(i)).add(zeroPadLenToken);
    return new PaddedTensor(paddedTensor.paddedTensor, lengths);
  }

  public PaddedTensor process(List batch) {
    List<Object> preprocessedBatch = new ArrayList<>();
    for (Object example: batch) {
      preprocessedBatch.add(this.preprocess(example));
    }
    return this.pad(preprocessedBatch);
  }

  public Object fillZeros(Object tensor) {
    if (tensor.getClass() == Integer.class)
      return 0;
    else {
      assert (tensor.getClass() == ArrayList.class);
      List<Object> zeroTensor = new ArrayList<>();
      for (Object example: (List)tensor) zeroTensor.add(fillZeros(example));
      return zeroTensor;
    }
  }

  public int size(Object x) {
    if (x.getClass() == String.class)
      return ((String)x).length();
    else {
      assert (x.getClass() == ArrayList.class);
      return ((List)x).size();
    }
  }

}
