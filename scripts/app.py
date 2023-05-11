import streamlit as st
import pandas as pd
import plotly.graph_objects as go

eda_matcher = {
    "Distribution of genres": {
        "df": "https://huggingface.co/spaces/AlekseyKorshuk/big-data-project/raw/main/output/eda/q1.csv",
        "description": "This query analyzes the distribution of genres across the entire Movie Lens dataset. It gives "
                       "us an overview of the most and least common movie genres, which can be useful for "
                       "understanding the overall makeup of the dataset. This could be a key insight if we consider "
                       "that user preferences for genres could significantly influence their movie ratings. "
                       "Therefore, understanding the genre distribution can help us fine-tune our recommendation "
                       "system and potentially highlight genre biases in the dataset."
    },
    "Average rating per movie": {
        "df": "https://huggingface.co/spaces/AlekseyKorshuk/big-data-project/raw/main/output/eda/q2.csv",
        "description": "This query calculates the average rating for each movie in the dataset. This information is "
                       "crucial for understanding how well-received each movie is overall. A movie with a higher "
                       "average rating is generally more favored by the audience, and such movies can be prioritized "
                       "in recommendations. However, it's also important to consider the number of ratings a movie "
                       "has received, as a high average rating based on a small number of reviews might not be as "
                       "reliable as a slightly lower average rating based on a large number of reviews."
    },
    "Number of ratings per movie": {
        "df": "https://huggingface.co/spaces/AlekseyKorshuk/big-data-project/raw/main/output/eda/q3.csv",
        "description": "This query determines the number of ratings each movie in the dataset has received. This is "
                       "important because it gives us an idea of a movie's popularity and the reliability of its "
                       "average rating. A movie with a large number of ratings is likely to have a more reliable "
                       "average rating. Moreover, highly-rated movies that have also received a lot of ratings can be "
                       "considered popular favorites and could be given higher priority in a recommendation system."
    },
    "Average rating of movies per genre": {
        "df": "https://huggingface.co/spaces/AlekseyKorshuk/big-data-project/raw/main/output/eda/q4.csv",
        "description": "This query computes the average rating for each genre in the dataset. This information can "
                       "help us understand which genres are generally more well-received by the audience. A genre "
                       "with a higher average rating might be more favored, and movies from these genres could be "
                       "recommended more often. It is important to remember that individual preferences can vary "
                       "widely, so genre preferences should be combined with other user-specific information to make "
                       "accurate recommendations."
    },
    "Top 10 movies with the highest average rating and at least 100 ratings": {
        "df": "https://huggingface.co/spaces/AlekseyKorshuk/big-data-project/raw/main/output/eda/q5.csv",
        "description": "This query lists the top 10 movies that not only have the highest average ratings but also "
                       "have received at least 100 ratings. The 100 ratings threshold helps ensure the reliability of "
                       "the average rating, as it filters out movies that might have a high average rating based on a "
                       "small number of ratings. The movies identified by this query could be considered highly "
                       "recommended by users, and the recommendation system might prioritize suggesting these movies "
                       "to users with similar tastes."
    },
}

for key, value in eda_matcher.items():
    eda_matcher[key]["df"] = pd.read_csv(value["df"], on_bad_lines='skip', sep="|" if "q5" in value["df"] else ",")

als_features_predictions = pd.read_csv(
    "https://huggingface.co/spaces/AlekseyKorshuk/big-data-project/raw/main/output/als_features_predictions.csv",
    index_col=0)
als_predictions = pd.read_csv(
    "https://huggingface.co/spaces/AlekseyKorshuk/big-data-project/raw/main/output/als_predictions.csv", index_col=0)

if __name__ == "__main__":
    st.title('Big Data: Movie RecSys')
    st.markdown(
        """Goal of this project is to develop a movie recommendation system that can predict individual user movie 
        ratings. This personalized recommendation system is expected to enhance the user experience by suggesting 
        movies based on their predicted preferences."""
    )

    st.header('Data Characteristics')
    st.markdown(
        """The datasets describe ratings and free-text tagging activities from MovieLens, a movie recommendation 
        service. It contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were 
        created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on October 
        17, 2016. Users were selected at random for inclusion. All selected users had rated at least 20 movies. No 
        demographic information is included. Each user is represented by an id, and no other information is provided. 

The data consists of 5 groups:

* rating.csv: contains ratings of movies by users:
    * userId
    * movieId
    * rating
    * timestamp
* movie.csv: contains movie information:
    * movieId
    * title
    * genres
* tag.csv: contains tags applied to movies by users:
    * userId
    * movieId
    * tag
    * timestamp
* genome_scores.csv: contains movie-tag relevance data:
    * movieId
    * tagId
    * relevance
* genome_tags.csv: contains tag descriptions:
    * tagId
    * tag
"""
    )

    st.header('Exploratory Data Analysis | Data Insights')
    eda_name = st.selectbox("Select query:", eda_matcher.keys())
    eda_dataframe = eda_matcher[eda_name]["df"]
    eda_description = eda_dataframe.describe()
    st.markdown(eda_matcher[eda_name]["description"])
    st.dataframe(eda_dataframe, use_container_width=True)
    st.dataframe(eda_description, use_container_width=True)

    st.header('Model Description')
    st.subheader('ALS: without features')
    st.markdown(
        """Second model doesn't use additional features like movie genres for prediction. It only uses 'userid', 
        'movieid', and 'rating'. Like in the first case, the 'prediction' column shows the ratings predicted by the 
        model."""
    )
    st.subheader('ALS: with features')
    st.markdown(
        """This model has taken into account the 'genres' of the movies along with other data like 'userid', 
        'movieid', and 'rating' to predict the rating a user would give a movie. The 'prediction' column contains the 
        predicted ratings generated by the ALS model, given the features. """
    )

    st.header('Model Performance')
    st.markdown(
        """We used two metrics to evaluate our models are Mean Average Precision (mAP) and Normalized Discounted Cumulative Gain (NDCG). 

1. Mean Average Precision (mAP): This is a metric used to evaluate the average of the precision scores at each rank for the list of movie recommendations. It's particularly useful when the order of the results matters, as in a recommendation system.

2. Normalized Discounted Cumulative Gain (NDCG): This metric accumulated from the top of the list to the bottom, with the gain of each result discounted at lower ranks. In other words, we prefer relevant items to appear earlier in the list rather than later. 

From our results, it seems that the ALS model with movie features (genres) performs better, both in terms of mAP and NDCG. This could be due to the additional information provided by the genres, which may allow the model to make more accurate recommendations, particularly for users who have strong preferences for certain genres."""
    )
    fig = go.Figure(
        data=[
            go.Bar(name='ALS with movie features', x=["mAP", "NDCG"], y=[0.861, 0.988]),
            go.Bar(name='ALS without features', x=["mAP", "NDCG"], y=[0.834, 0.973]),
        ]
    )
    st.plotly_chart(fig)

    st.header('Prediction Results')
    st.markdown("TODO")
    st.subheader('ALS: with movie features')
    st.dataframe(als_features_predictions, use_container_width=True)
    st.subheader('ALS: without movie features')
    st.dataframe(als_predictions, use_container_width=True)

    st.header('Conclusion')
    st.markdown(
        """We made an analysis of MoveLens dataset and experimented with 2 different approaches in model development:
        vanilla ALS and ALS with movie features. We evaluated our models on 30% test split with mAP and NDCG. 
        As a result, ALS with movie features performs way better and archives good results for both metrics."""
    )

    st.header('About')
    st.markdown("""
    * GitHub: [![GitHub Repo stars](https://img.shields.io/github/stars/rafailvv/bigdata-final-project-iu-2023?style=social)](https://github.com/rafailvv/bigdata-final-project-iu-2023)
    * Authors:
        * [Aliaksei Korshuk](https://github.com/AlekseyKorshuk)
        * [Rafail Venediktov](https://github.com/rafailvv)
    * We would like to thank Professor and TA for this course.
    """)
