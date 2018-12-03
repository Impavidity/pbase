package botafogo;

import org.apache.jena.query.*;
import org.apache.jena.tdb.base.file.Location;
import org.apache.jena.tdb.TDBFactory;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.OptionHandlerFilter;
import org.kohsuke.args4j.ParserProperties;

import java.io.File;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;


public class TDBQuery {
  private static final Logger LOG = LogManager.getLogger(TDBQuery.class);

  private Location location;
  private String directoryPath;
  private Dataset global_dataset;

  AtomicInteger cnt;

  public TDBQuery(String directoryPath) {
    LOG.info(String.format("Index Location : %s", directoryPath));
    this.directoryPath = directoryPath;
    this.location = Location.create(directoryPath);
    this.global_dataset = TDBFactory.createDataset(location);
    LOG.info("Create global dataset");
  }

  private final class QueryCallable implements Callable<List<List<QuerySolution>>> {
    final private String[] queries;
    final private Dataset dataset;

    QueryCallable(Location location, String[] queries) {
      this.dataset = TDBFactory.createDataset(location);
      this.queries = queries;
    }

    @Override
    public List<List<QuerySolution>> call() {
      List<List<QuerySolution>> results = new ArrayList<>();
      dataset.begin(ReadWrite.READ);
      try {
        for (String query: queries) {
          List<QuerySolution> result = new ArrayList<>();
          try(QueryExecution qExec = QueryExecutionFactory.create(query, dataset)) {
            ResultSet rs = qExec.execSelect();
            while (rs.hasNext())
              result.add(rs.next());
          } catch (Exception e) {
            LOG.info(query);
            LOG.error(e);
          }
          results.add(result);
          int cur = cnt.incrementAndGet();
          if (cur % 500 == 0) {
            LOG.info(cnt + " queries issued");
          }
        }
      } finally {
        dataset.end();
        dataset.close();
      }
      return results;
    }
  }

  public List<String[]> chunkQueries(String[] queries, int numChunks) {
    List<String[]> chunkedQueries = new ArrayList<>();
    int chunkSize = (int) Math.ceil((double) queries.length / numChunks);
    LOG.info("Chunk size of parallel query is " + chunkSize);
    for (int i = 0; i < queries.length; i+=chunkSize)
      chunkedQueries.add(Arrays.copyOfRange(queries, i, Math.min(queries.length,i+chunkSize)));
    LOG.info("There are " + chunkedQueries.size() + " in total");
    return chunkedQueries;
  }

  private List<List<QuerySolution>> getFuture(Future<List<List<QuerySolution>>> future) {
    try {
      return future.get();
    } catch (InterruptedException e) {
      e.printStackTrace();
    } catch (ExecutionException e) {
      e.printStackTrace();
    }
    return new ArrayList<>();
  }

  public List<List<QuerySolution>> parallelQuery(String[] queries, int numThreads) {
    // The number of threads used in this process is not exactly the same as argument.
    // The real threads number depends on the chunking size.
    final ExecutorService executor = Executors.newFixedThreadPool(numThreads);
    List<String[]> chunkedQueries = chunkQueries(queries, numThreads);
    List<Future<List<List<QuerySolution>>>> futures = new ArrayList<>();
    int numChunks = chunkedQueries.size();
    cnt = new AtomicInteger();
    for (int i = 0; i < numChunks; i++) {
      Future<List<List<QuerySolution>>> future = executor.submit(new QueryCallable(location, chunkedQueries.get(i)));
      futures.add(future);
    }
    try {
      return futures.stream().flatMap(future -> getFuture(future).stream()).collect(Collectors.toList());
    } finally {
      executor.shutdown();
    }
  }

  public List<List<QuerySolution>> sequentialQuery(String[] queries) {
    List<List<QuerySolution>> results = new ArrayList<>();
    global_dataset.begin(ReadWrite.READ);
    int cnt = 0;
    try {
      for (String query: queries) {
        List<QuerySolution> result = new ArrayList<>();
        try(QueryExecution qExec = QueryExecutionFactory.create(query, global_dataset)) {
          ResultSet rs = qExec.execSelect();
          while (rs.hasNext())
            result.add(rs.next());
        } catch (Exception e) {
          LOG.info(query);
          LOG.error(e);
        }
        results.add(result);
        cnt += 1;
        if (cnt % 500 == 0) {
          LOG.info(cnt + " queries issued");
        }
      }
    } finally {
      global_dataset.end();
      global_dataset.close();
    }
    return results;
  }

  public List<QuerySolution> query(String query) {
    global_dataset.begin(ReadWrite.READ);
    List<QuerySolution> result = new ArrayList<>();
    try(QueryExecution qExec = QueryExecutionFactory.create(query, global_dataset)) {
      ResultSet rs = qExec.execSelect();
      while (rs.hasNext())
        result.add(rs.next());
    } catch (Exception e) {
      LOG.info(query);
      LOG.error(e);
    } finally {
      global_dataset.end();
    }
    return result;
  }

  public void close() {
    this.global_dataset.close();
    String journalPath = Paths.get(this.directoryPath, "journal.jrnl").toString();
    File journal = new File(journalPath);
    if (!journal.delete())
      LOG.info("The journal deos not removed");
  }

  static final class Args {
    @Option(name = "-index", metaVar = "[path]", required = true, usage = "index path")
    public String index;
    @Option(name = "-query", metaVar = "[query]", required = true, usage = "sparql query")
    public String query;

  }

  public static void main(String[] args) throws Exception {
    Args queryArgs = new Args();
    CmdLineParser parser = new CmdLineParser(queryArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: "+ TDBQuery.class.getSimpleName() +
              parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }

    TDBQuery tdbQuery = new TDBQuery(queryArgs.index);
    String q1 = queryArgs.query;
    try {
      List<QuerySolution> resultSet = tdbQuery.query(q1);
      for (QuerySolution result: resultSet) {
        Iterator<String> iter = result.varNames();
        while (iter.hasNext())
          System.out.print(result.get(iter.next()) + "\t");
        System.out.println();
      }
      List<String> queries = new ArrayList<>();
      for (int i = 0; i < 10; i ++)
        queries.add(q1);
      String[] queryStrings = queries.toArray(new String[0]);
      List<List<QuerySolution>> resultSets = tdbQuery.parallelQuery(queryStrings, 4);
      for (List<QuerySolution> eachResults: resultSets) {
        for (QuerySolution result: eachResults) {
          Iterator<String> iter = result.varNames();
          while (iter.hasNext())
            System.out.print(result.get(iter.next()) + "\t");
          System.out.println();
        }

      }
    } finally {
      tdbQuery.close();
    }


  }
}
