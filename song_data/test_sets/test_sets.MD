# TEST SETS
We generated test sets from the original data by introducing various types of alterations to the data. The resulting data are therefore invalid song lyrics. Following are descriptions of each modification.

## Shuffled 
Lines of songs are shuffled randomly (using python's ramdom.sample()). There is no additional correction - if two newlines end up next to each other they stay that way, otherwise it would alter the line count of the song.

## Half-shuffled
Like "shuffled" only shuffles the first half of all lines. Randomly shuffles indexes for the first half of the lines and then assigns to each line the verse with the selected index.

## Replaced word
Each line of the lyrics is split to words on space character. For each word a random number between 0 and 1 is generated. If this number smaller than given probability (default = 0.33) then the corresponding word is replaced with selected word (default = "giraffe").

## Shift noun back
Applies Python's POS tagger for each line to assign tag to each token on the line. Then each line is looked at word by word from left to right. When 'NN' tag (= singular noun) is encountered it's swaped with preceding word. Swaping does not occur between lines.