import click
from tqdm import tqdm

from pbase.ptext.vocab import (
  vocab_vector_file_parser,
  vocab_file_parser)


@click.group()
def cli():
  pass

@click.command()
@click.option("--vocab_file", type=str, required=True)
@click.option("--embeddings", type=str, required=True)
@click.option("--lower", type=bool, default=True)
@click.option("--output_file", type=str, required=True)
def filter_embedding(vocab_file, embeddings, lower, output_file):
  embedding_vector, embedding_size, embedding_dim = vocab_vector_file_parser(embeddings, lower)
  vocab, vocab_size, = vocab_file_parser(vocab_file)
  with open(output_file, "w") as fout:
    for token in tqdm(vocab.keys()):
      if token in embedding_vector:
        embedding = embedding_vector[token]
        fout.write("{} {}\n".format(token, " ".join([str(f) for f in embedding])))
  click.echo("Writing embedding to {}".format(output_file))

cli.add_command(filter_embedding)


if __name__ == "__main__":
  cli()