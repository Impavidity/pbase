package pbase.ptext;


import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


public class Vocab {
  Map<Integer, String> itos = new HashMap<>();
  Map<String, Integer> stoi = new HashMap<>();
  String vocabFilePath;
  int unkTokenIdx;

  public Vocab(String vocabFilePath, String unkToken) throws Exception {
    BufferedReader br = new BufferedReader(new FileReader(vocabFilePath));
    String st;
    String[] numberToken;
    Integer id;
    String token;
    while ((st = br.readLine()) != null) {
      numberToken = st.split(" ", 2);
      id = Integer.valueOf(numberToken[0]);
      token = numberToken[1];
      itos.put(id, token);
      stoi.put(token, id);
    }
    unkTokenIdx = stoi.getOrDefault(unkToken, -1);
    this.vocabFilePath = vocabFilePath;
  }

  public int stringToIndex(String token) {
    return stoi.getOrDefault(token, unkTokenIdx);
  }

  public List<Integer> batchToIndex(List<String> batch) {
    return batch.stream().map(str -> stringToIndex(str)).collect(Collectors.toList());
  }

  public String indexToString(int index) {
    return itos.get(index);
  }

  public List<String> batchToString(List<Integer> batch) {
    return batch.stream().map(index -> indexToString(index)).collect(Collectors.toList());
  }
}
