import click

@click.command()
@click.option("--name", prompt="Your name", help="The person to greet.")
@click.option("--id", prompt="Your ID", help="The person's id")
def hello(name,id):
    click.echo(f"Hello, {name}-{id}!")

if __name__ == '__main__':
    print('?')
    hello(name='test',id=6339)
    hello()
