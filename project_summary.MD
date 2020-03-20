# Introduction
This markdown file serves to communication my experiences on this project.

## Project Description
As everything, this project changed a little during development. The original description is below:

### Original Description
Extend sklearn's NMF class to incorporate functions from nimfa.

- Input: a (MxN) matrix.
- Output: two matrices (W, H) whose product is approximately the input matrix.
- Justification: this problem is a software engineering problem. I would isolate the necessary classes in from the sklearn library and extend the class. I would include simulation examples and tests.

Comments:
  _"The `sklearn` implementation of NMF provides a very sleek interface, but at the expense of burying a large amount of information in a single function, with no convenient way to to access it. I would add intermediate output to the `sklearn` class, modeled after the `nifma` class. Specifically, I would add feature selection tools to the NMF class to allow for efficient manipulation (like the PCA class in sklearn. In my research, this would enable me to check my work more thoroughly, to explain my models better as I write manuscripts, and to combine/compare approaches more easily."_    

### Updates
The package `nimfa` and `sklearn` both offer NMF methods. The `nimfa` implementation is significantly more advanced and flexible, but does not "play well" with the larger, more popular `sklearn`. My original goal was to "merge" these two functionalities by adding more advanced functionality to the `sklearn` class. This proved to be out-of-scope due to the complexity and interdependence of the NMF methods in the `nimfa`. For example, `nimfa` contains a [custom linear algebra module](https://github.com/marinkaz/nimfa/blob/master/nimfa/utils/linalg.py).

As a result, my focus shifted from merging two libraries to extending the `NMF` class from `sklearn` in order to add functionality missing (or scattered) from both `nimfa` and `sklearn`. Specifically, I have chosen to make the analysis of the reconstruction much friendlier, and easier to use. Enjoy!

## Questions
### What was your biggest challenge in this project?
Organization was the biggest challenge for me. In order to make a sleek interface for `NMFE` I had to make sure class methods had clearly defined scopes and parameters. I had to find ways to re-use as much of the code as I could without needlessly sacrificing flexibility. For example, I found it challenging to manage the ability to 'refit' a factorization and allow the user the ability to call multiple factorizations on the same data without re-writing code and without blindly overwriting class attributes.

The other thing I found challenging was making decisions for the end-user. I was not sure which procedures made sense to automate vs demand a function call for. It was not immediately obvious, for example, whether normalizing the input matrix should demand a separate, purposeful function call the way the reshaping the matrix does. This challenge also included (to some degree), picking default values. My motivation was to provide the simple, expected results from most procedures and allow the user to add bells and whistles as they see fit.

In a few places I broke Python conventions. When computing fit error, for example. The method  `NMFE.compute_fit_error` returns a dictionary, not a floating point value. This is akin to the R community, where output is generally well labeled. My choice here was primarily to make sure that the user knows what they are getting. It's pretty easy to forget during a workflow...

Another time I broke convention is with the `NMFE.compute_hausdorff()` function which returns objects of different types depending on the input parameters. This, I think, makes sense...

Finally - there is the `NMFE.to_square()` method which breaks a few rules. This method does not generally return anything. This method is used to reshape the input matrix and is the only instance when `NMFE` prints a message to the screen. For now, this cannot be silenced (though this is an easy change). The reason this is dangerous is that it is not immediately clear if you have overwritten your class attributes unless there is a message. Still, having the ability to direct the output of this allows one to compare `NMFE` instances against one another. For example, it is easy enough to chain these objects together:

```Python
A, B, C, D = nmfe.refit(in_place=False)
nmfe_sq = NMFE(input_matrix=A,
               n_components = nmfe._n_components)
```

So basically, in addition to organizing my thoughts in a functioning manner, finding the right balanced os assumptions was also difficult. I also feel I should mention that I am terrible at waiting long enough to refactor. As a result I often rename things locally and have a little nightmare when testing.

### What did you learn while working on this project?
I learned many things. I have broken the into little "groups" of lessons below:

- `__init__` ordering: I learned about the difficultly of initialization ordering. I had not considered how tricky it can be to establish the correct ordering of procedures when constructing the class. How should user inputs be validated? What should be baked into the `__init__()` function vs moved elsewhere?
- testing: One always learns a lot about dependencies and ordering when testing. I learned about how to mitigate different invalid options and handle errors using something that could help the user get to the correct answer.
- linear algebra: I know it's not the point of this course, but I had re-learn some things. For example, which distances make sense for matrices?
- I learned a lot about class structure. I have used classes before to accomplish things, but this was fun to think about designing and testing a class object. It for sure made me a better programmer.
- This lesson: "if it's in two places it might be because they don't play well together". It's possible people have entertained merging `nimfa` and `sklearn` or building an API between them before. But man, this would require a complete overhaul of at least one, if not both libraries.

### I you had more time on the project what other question(s) would you like to answer?
Ha. So this implementation is cute, but non-robust. Other obvious questions include:

1. Is it possible to make a `base_factorization` evaluator that can perform on a general set of factorizations? Things that come to mind are SVD, PCA, Bayesian NMF, MNMF, ect. This line of thinking is valid not just for factorization, but in general scientific software doesn't promote tinkering as much as some (me) would like. So the question is: is it possible to inspire academic play with mathematical objects?
2. Can `nimfa` and `sklearn` libraries be effectively combined? I believe so, but don't have a year to do it.
3. Can _more_ functionality be added? Yes. With more time I would dive deeper into rudimentary linear algebra operations. It also strikes me that the Python community has linear algebra tools poorly organized (split between `numpy` and `scipy`). Alas, a challenge for another day...
4. Simulation ability. I would have liked to do some simulation functions. I find these incredibly helpful for understanding the applicable ranges of things and the essential relationships between things.
5. More mathematical checks. These are hasty functions with no theoretical checks. More time = better math.

I could probably go on, but I hope this is sufficient.
