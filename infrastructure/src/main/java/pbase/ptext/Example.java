package pbase.ptext;


import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Example {
  private Map<String, Object> variables = new HashMap<>();
  Map<String, Object> example;
  List<Attribute> attributes;


  public Example(Map example, List<Attribute> attributes) {
    this.example = example;
    this.attributes = attributes;
    for (Attribute attribute: attributes) {
      variables.put(attribute.target,
              attribute.field.preprocess(
                      example.get(attribute.source)));
    }
  }

  @Override
  public String toString() {
    return "Example:\n" +
            "Raw Data:\n" +
            this.example +
            "\nProcessed Data:\n" +
            String.join("\n", this.attributes.stream().map(
                    attribute -> attribute.target + ":\n" + this.variables.get(attribute.target)
            ).collect(Collectors.toList()));
  }

}
