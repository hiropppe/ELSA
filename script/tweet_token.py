import click
import json
import os

from pathlib import Path
from collections import Counter
from word_generator import TweetWordGenerator


@click.command()
@click.argument("input_path")
@click.argument("data_dir")
@click.argument("lang")
@click.option("--processes", "-w", default=os.cpu_count()-1, help="")
def main(input_path, data_dir, lang, processes):
    data_dir = Path(data_dir)
    token_output = (data_dir / '{:s}_tokens.txt'.format(lang)).__str__()
    emoji_output = (data_dir / '{:s}_emoji.txt'.format(lang)).__str__()

    emoji_wanted = json.loads(open("./emoji_unicode", "r").read()).keys()

    wg = TweetWordGenerator(input_path,
                            lang,
                            emojis=emoji_wanted,
                            processes=processes)

    n_sents, n_emoji, n_emoji_sents = 0, 0, 0
    emoji_freq = Counter()

    with open(token_output, 'w') as fot:
        for (tokens, info) in wg:
            n_sents += 1
            emojis = info["emojis"]
            if emojis:
                n_emoji += len(emojis)
                n_emoji_sents += 1
                for emoji in emojis:
                    emoji_freq[emoji] += 1
            print(' '.join(tokens), file=fot)

    print("{:d} sents {:d} emojis {:d}({:.3f}%) emoji_sents".format(
        n_sents, n_emoji, n_emoji_sents, float(n_emoji_sents)*100/n_sents))

    with open(emoji_output, "w") as foe:
        for token, freq in emoji_freq.most_common(n=len(emoji_freq)):
            print("{:s}\t{:d}".format(token, freq), file=foe)


if __name__ == '__main__':
    main()
