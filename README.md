# fontshot

This is part of a broader effort to teach neural networks to do one-shot learning. The idea is that, if you show a neural net a bunch of letters in the alphabet, it should be able to quickly learn the remaining letters.

To collect alphabet data, I got a bit creative. I wrote a tool called [font-dump](https://github.com/unixpickle/font-dump) to fetch hundreds of English fonts. The tool dumps these fonts as image files, with one file per character per font. The result is that we have 62 classes (numerals and both cases of letters).

There are two parts of the model: a learner and a classifier. The learner takes a few examples of a character, and the classifier takes knowledge from the learner to classify new characters. In particular, the learner generates a small vector which the executor uses as an internal representation of the learned character.

# Related work

After having the idea, I searched around and found [omniglot](https://github.com/brendenlake/omniglot). That dataset seems to be inspired by a *very* similar idea to the one I had, although they gathered data in a more conventional (possibly better) way.
