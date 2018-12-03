package zoo.ner;

import com.google.common.primitives.Ints;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.SavedModelBundle;
import pbase.ptext.Vocab;

import java.util.Arrays;
import java.util.List;
import java.nio.IntBuffer;


public class NER {
  public static void main(String[] args) throws Exception {
    SavedModelBundle bundle = SavedModelBundle.load("saves", "serve");
    Session sess = bundle.session();
    Vocab vocab = new Vocab("data/word_vocab.txt", "<unk>");
    String sentence = "barack obama visited golden bridge in san francisco";
    List<String> tokens = Arrays.asList(sentence.split(" "));
    List<Integer> indexs = vocab.batchToIndex(tokens);
    long[] shape = {1L, 8L};
    Tensor wordTensor = Tensor.create(shape, IntBuffer.wrap(Ints.toArray(indexs)));
    System.out.println(wordTensor);


  }
}
