import click

from pbase.pdata import OntonotesNamedEntityRecognition


@click.group()
def cli():
  pass


@click.command()
@click.option('--file_path', type=str)
@click.option('--dump_path', type=str)
def ontonotes_ner(file_path, dump_path):
  ontonotes_ner = OntonotesNamedEntityRecognition()
  ontonotes_ner.dump(file_path, dump_path)


cli.add_command(ontonotes_ner)


if __name__ == "__main__":
  cli()