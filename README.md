# fontshot

This is part of a broader effort to teach neural networks to do one-shot learning. The idea is that, if you show a neural net a bunch of letters in the alphabet, it should be able to quickly learn the remaining letters.

To collect alphabet data, I got a bit creative. I wrote a tool called [font-dump](https://github.com/unixpickle/font-dump) to fetch hundreds of English fonts. The tool dumps these fonts as image files, with one file per character per font. The result is that we have 62 classes (numerals and both cases of letters). We can make 4*62=248 classes out of this data by rotating the samples at 90 degree increments.

Right now, the model is fairly symmetric. It takes two samples and tries to predict whether or not they belong to the same class.

# Related work

After having the idea, I searched around and found [omniglot](https://github.com/brendenlake/omniglot). That dataset seems to be inspired by a *very* similar idea to the one I had, although they gathered data in a more conventional (possibly better) way.
