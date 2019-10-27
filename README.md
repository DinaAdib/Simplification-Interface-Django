# Text Simplifcation and Summarization System
This project has been developed for our bachelor's graduation project. 
It's a system that simplifies English Text on both the lexical and syntactic levels and also summarizes it.
For **Syntactic Simplification**, it utilizes RNN trained on Newsela Dataset , while **Lexical Simplification** is based on a Machine Learning Model trained on BenchLS. The **summarizer** is rule-based.

## Sample System Output: 

### Syntactic Simplification 
**_Input_** A succession of steadily more powerful and flexible computing devices were constructed in the 1930s and 1940s , gradually adding the key features that are seen in modern computers.
**_Output_** More powerful and simpler computers were made in the 1930s and 1940s .


### Lexical Simplification 
**_Input_** When the leg is released , the popcorn moves like an gymnast somersaulting.
**_Output_** When the leg is released , the popcorn moves like an athlete flip.



### Summarizer
**_Input_**  'Alcohol' is taken in almost all cool and cold climates, and to a very much less extent in hot ones. Thus, it is taken by people who live in the Himalaya Mountains, but not nearly so much by those who live in the plains of India. Alcohol is not necessary in any way to anybody. The regular use of alcohol, even in small quantities, tends to cause mischief in many ways to various organs of the body. It affects the liver, it weakens the mental powers, and lessens the general energy of the body. In addition, damage to the central nervous system and peripheral nervous system can occur from chronic alcohol abuse.
**_Output_** Alcohol' is taken in almost all cool and cold climates, and to a very much less extent in hot ones. Alcohol is not necessary in any way to anybody. The regular use of alcohol, even in small quantities, tends to cause mischief in many ways to various organs of the body.

