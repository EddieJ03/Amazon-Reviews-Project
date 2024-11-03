# 151A NLP Project

## Milestone 2

1. For missing/null titles and text: 
    - If for an observation the title is missing but not the text, we will fill the title with the text 
    - If for an observation the text is missing but not the title, we will fill the text with the title
    - If both are missing:
        - if the rating is 4 or 5, we will replace both both title and text with the phrase "Very good"
        - if the rating is 3, we will replace both both title and text with the word "Good"
        - if the rating is 1 or 2, we will replace both both title and text with the word "Bad"

2. For all titles and text, we plan to clean the string in the following way:
    - Remove HTML entities and tags
    - Remove emojis
    - Convert all characters to lowercase
    - Remove all leading, trailing, and extra spaces between words

3. Only keep necessary columns:
    - We really only plan to keep the following columns: rating, title, and text

4. Filter out unverified purchases:
    - For out purposes, we will only focus on purchases that our verified so we can have more trust in the rating