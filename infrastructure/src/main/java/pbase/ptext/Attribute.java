package pbase.ptext;

public class Attribute {
  String target;
  String source;
  Field field;

  public Attribute(String target, String source, Field field) {
    this.target = target;
    this.source = source;
    this.field = field;
  }
}
