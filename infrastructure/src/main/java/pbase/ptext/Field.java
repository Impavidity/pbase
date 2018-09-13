package pbase.ptext;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Field {

  boolean sequential;
  Object initToken;
  Object eosToken;
  boolean lower;
  Function<String, List> tokenize;
  Object padToken;
  int fixLength;

  public Field(
          boolean sequential,
          Object initToken,
          Object eosToken,
          boolean lower,
          Function<String, List> tokenize,
          Object padToken,
          int fixLength) {
    this.sequential = sequential;
    this.initToken = initToken;
    this.eosToken = eosToken;
    this.lower = lower;
    this.tokenize = tokenize;
    this.padToken = padToken;
    this.fixLength = fixLength;
  }

  public Object preprocess(Object example) {
    if (this.sequential && example.getClass() == String.class) {
      example = this.tokenize.apply((String)example);
    }
    if (this.lower) {
      example = ((List<String>)example).stream()
              .map(token -> token.toLowerCase())
              .collect(Collectors.toList());
    }
    return example;
  }

  public PaddedTensor pad(List batch) {
    int maxLen = 0;
    if (!this.sequential) {
      List<Integer> lengths = new ArrayList<>();
      for (int i=0; i<batch.size(); i++)
        lengths.add(1);
      return new PaddedTensor(batch, lengths);
    }

    if (this.fixLength < 0) {
      for (Object example: batch) {
        int len = ((List)example).size();
        if (len > maxLen) maxLen = len;
      }

    } else {
      maxLen = this.fixLength -
              ((this.initToken  == null) ? 0 : 1) -
              ((this.eosToken == null) ? 0 : 1);
    }
    List<Object> paddedList = new ArrayList<>();
    List<Integer> lengths = new ArrayList<>();
    for (Object example: batch) {
      List<Object> padded = new ArrayList<>();
      if (this.initToken != null) padded.add(this.initToken);
      padded.addAll(((List) example).subList(0, Math.min(maxLen, ((List) example).size())));
      if (this.eosToken != null) padded.add(this.eosToken);
      int exampleLength = ((List) example).size();
      int paddingLimit = Math.max(0, maxLen - exampleLength);
      for (int repeat = 0; repeat < paddingLimit; repeat++)
        padded.add(this.padToken);
      paddedList.add(padded);
      lengths.add(padded.size() - paddingLimit);
    }
    return new PaddedTensor(paddedList, lengths);
  }

  public PaddedTensor process(List batch) {
    List<Object> preprocessedBatch = new ArrayList<>();
    for (Object example: batch) {
      preprocessedBatch.add(this.preprocess(example));
    }
    return this.pad(preprocessedBatch);
  }
}
