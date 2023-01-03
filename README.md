# Shakespeare data

This repo contains plain text versions of the complete works of Shakespeare, copied & pasted from this website: [http://shakespeare.mit.edu/]{http://shakespeare.mit.edu/}. The html from that website as well as the plain text copies included here are considered in the public domain. 

The plays and poetry are separated into the respective directories `plays` and `poetry`. 

## Installation

This repo was created with `git lfs`. If you do not already have `git lfs` installed, run:

`git lfs install`

After that, you should be able to clone the repository as per usual. 


## Preprocessing script

The `preprocessing.py` script may be used to prepare the plain text files for language modeling, for dialogue. 
NOTE: At this time, preprocessing is included only for the play dialogues. No prep has yet been done for preprocessing the poetry for language modeling.

Some useful functions included there:

- `text_to_dialogues()`: takes a directory of text files (like the `plays` dir) and produces a new directory of jsonlines dialogues, in the format: 

```
{"character_and_line": "COUNTESS\nIn delivering my son from me, I bury a second husband.\n"}
{"character_and_line": "BERTRAM\nAnd I in going, madam, weep o'er my father's death\nanew: but I must attend his majesty's command, to\nwhom I am now in ward, evermore in subjection.\n"}
{"character_and_line": "LAFEU\nYou shall find of the king a husband, madam; you,\nsir, a father: he that so generally is at all times\ngood must of necessity hold his virtue to you; whose\nworthiness would stir it up where it wanted rather\nthan lack it where there is such abundance.\n"}
```

The new directory will be broken down into sub-directories per play, with numbered `dialogue_N.json` files.

- `dialogue_to_groups()`: takes a directory of dialogue jsonlines and a tokenizer (from the `transformers` library), along with a maximum input length, and groups dialogue into seq2seq pairs of an ever-growing previous context, up to the maximum input token length, truncating from the beginning of the previous context. For a concrete example, the above three lines from All's Well That Ends Well would become two pairs of dialogue sequences:

```
{"taskname": "eme-seq2seq", "context": "COUNTESS\nIn delivering my son from me, I bury a second husband.\n", "response": "BERTRAM\nAnd I in going, madam, weep o'er my father's death\nanew: but I must attend his majesty's command, to\nwhom I am now in ward, evermore in subjection.\n"}
{"taskname": "eme-seq2seq", "context": "COUNTESS\nIn delivering my son from me, I bury a second husband.\nBERTRAM\nAnd I in going, madam, weep o'er my father's death\nanew: but I must attend his majesty's command, to\nwhom I am now in ward, evermore in subjection.\n", "response": "LAFEU\nYou shall find of the king a husband, madam; you,\nsir, a father: he that so generally is at all times\ngood must of necessity hold his virtue to you; whose\nworthiness would stir it up where it wanted rather\nthan lack it where there is such abundance.\n"}
``` 

If the context triggers the maximum input length, then it would be truncated by a character's utterance, starting from the beginning, yielding the following for the second dialogue pair:
```
{"taskname": "eme-seq2seq", "context": "BERTRAM\nAnd I in going, madam, weep o'er my father's death\nanew: but I must attend his majesty's command, to\nwhom I am now in ward, evermore in subjection.\n", "response": "LAFEU\nYou shall find of the king a husband, madam; you,\nsir, a father: he that so generally is at all times\ngood must of necessity hold his virtue to you; whose\nworthiness would stir it up where it wanted rather\nthan lack it where there is such abundance.\n"}
```

- `create_train_and_val_directories()`: using the token lengths of scenes and prepared dialogues, divides the plays into train and validation directories. 