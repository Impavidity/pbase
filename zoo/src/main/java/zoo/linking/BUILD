java_library(
  name='candidateRetrieval',
  sources='CandidateRetrieval.java',
  dependencies = [
   '3rdparty/jvm:anserini',
   '3rdparty/jvm:lucene',
  ],
)

jvm_binary(name = "linking",
  source = 'Linking.java',
  main = 'zoo.linking.Linking',
  dependencies = [
    'zoo/src/main/java/zoo/linking:candidateRetrieval',
  ],
)
