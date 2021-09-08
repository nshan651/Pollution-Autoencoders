# Meeting 1
# Paper Creation Section 1

   - Divide up work
      - Intro/abstract, problem, methodology, biz case, results, related work/discussion
      - Someone should do the coding
   - Business case
   - Be specific about data cleaning/imputing (1-2 pages on methodology)
   - Add some nicely formatted graphs/features



#Meeting 07/20/21: 

## Methodology
- make co concentrationtable more readable, change font from black
- not much preprocessing was done
- dependent is part of the input
- "normalized the x values in the input data"
- reason why we picked scalar
    -The results are dramatically better. The difference is due to simple
     geometry. The initial hyperplanes pass fairly near the origin. If the data
     are centered near the origin (as with {-1,1} coding), the initial
     hyperplanes will cut through the data in a variety of directions. If the
     data are offset from the origin (as with {0,1} coding), many of the initial
     hyperplanes will miss the data entirely, and those that pass through the
    data will provide a only a limited range of directions, making it difficult
    to find local optima that use hyperplanes that go in different directions.
    If the data are far from the origin (as with {9,10} coding), most of the
    initial hyperplanes will miss the data entirely, which will cause most of
    the hidden units to saturate and make any learning difficult. See "Should I
    standardize the input variables?" for more information.
    
    -http://www.faqs.org/faqs/ai-faq/neural-nets/part2/
- justify the "Why"
- "we segregated the x and y in input data then we normalized x"
- feature addition of daily averages of hourly data
- write down thought process in notes
- change boxes
- change data pipeline so that ae/pca in box 3, linreg on encoded/unencoded,
  results of all three
- Add more details to methodology
- feature engineering/processing
- why did we take the mean of every day?
- why didn't we need to impute values?
- check if greater than 10% zeroes present in the values
- muted colors like blue/green scale in CO

## TODO:

✓ make co concentrationtable more readable, change font from black
- add more to preprocessing
- fix part about dependent var
- add to the reasoning in the methodology/abstracts. "Why?"
- write down thought process in notes
- Change pipeline graphic in box 3, linreg on encoded/encoded data
    - results of all three

## Methodology explaination

- make a covariance matrix for pca/ae
- talk about number of components/dimensions that were chosen for "optimal" explaination
    -Meaning, what was the fewest no. of dimensions that explained the most for each model
- Link to article explaination: https://psico.fcep.urv.cat/utilitats/factor/documentation/Percentage_of_explained_common_variance.pdf


## Meeting 07/30/2021







