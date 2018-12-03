package zoo.linking;

import io.anserini.analysis.FreebaseAnalyzer;
import io.anserini.kg.freebase.IndexFreebase;
import io.anserini.kg.freebase.LookupFreebase;
import io.anserini.rerank.ScoredDocuments;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class CandidateRetrieval {

  private static final Logger LOG = LogManager.getLogger(LookupFreebase.class);

  private final IndexReader reader;

  protected class Result {
    public String mid;
    public String name;
    public String wikiTitle;
    public String w3Label;
    public float score;

    public Result(String mid, String name, String wikiTitle, String w3Label, float score) {
      this.mid = mid;
      this.name = name;
      this.wikiTitle = wikiTitle;
      this.w3Label = w3Label;
      this.score = score;
    }
  }


  public CandidateRetrieval(Path indexPath) throws IOException {
    if (!Files.exists(indexPath) || !Files.isDirectory(indexPath) || !Files.isReadable(indexPath)) {
      throw new IllegalArgumentException(indexPath + " does not exist or is not a directory.");
    }

    this.reader = DirectoryReader.open(FSDirectory.open(indexPath));
  }

  public Result[] retrieve(String q) throws Exception {
    IndexSearcher searcher = new IndexSearcher(this.reader);
    MultiFieldQueryParser queryParser = new MultiFieldQueryParser(
            new String[]{ IndexFreebase.FIELD_NAME, IndexFreebase.FIELD_LABEL, IndexFreebase.FIELD_ALIAS },
            new FreebaseAnalyzer());
    queryParser.setDefaultOperator(QueryParser.Operator.OR);
    Query query = queryParser.parse(q);
    ConstantScoreQuery constantQuery = new ConstantScoreQuery(query);
    TopDocs rs = searcher.search(constantQuery, Integer.MAX_VALUE);
    ScoredDocuments docs = ScoredDocuments.fromTopDocs(rs, searcher);
    for (int i = 0; i < 10; i++) {
      System.out.println(docs.documents[i]);
    }
    return null;
  }
}
