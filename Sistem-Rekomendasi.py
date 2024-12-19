import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
DATA_RATING_PATH = 'DataRatingUser.xlsx'
DATA_WISATA_PATH = 'DataWisata.xlsx'

def load_data():
    rating_df = pd.read_excel(DATA_RATING_PATH)
    wisata_df = pd.read_excel(DATA_WISATA_PATH)
    # Convert 'fasilitas' column to lowercase
    wisata_df['fasilitas'] = wisata_df['fasilitas'].str.lower()
    return rating_df, wisata_df

rating_df, wisata_df = load_data()

# Function to calculate item similarity using cosine similarity
def calculate_item_similarity(rating_df):
    rating_matrix = rating_df.iloc[:, 4:24].fillna(0)  # Use only rating columns
    similarity = cosine_similarity(rating_matrix.T)
    return pd.DataFrame(similarity, 
                        index=rating_df.columns[4:24], 
                        columns=rating_df.columns[4:24])

# Function to calculate TF-IDF similarity for facilities
def get_tfidf_similarity(wisata_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(wisata_df['fasilitas'].fillna(''))
    similarity = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(similarity, 
                        index=wisata_df['wisata'], 
                        columns=wisata_df['wisata'])

# Recommend places based on selected wisata
def recommend_wisata(item_similarity, tfidf_similarity, wisata, top_n=5, alpha=0.6):
    if wisata not in item_similarity.index or wisata not in tfidf_similarity.index:
        raise ValueError(f"Wisata {wisata} tidak ditemukan dalam data.")

    combined_similarity = alpha * item_similarity[wisata] + (1 - alpha) * tfidf_similarity[wisata]
    recommendations = combined_similarity.sort_values(ascending=False).head(top_n + 1)
    recommendations = recommendations[1:]  # Exclude the selected wisata itself
    return recommendations

# Create a tab bar for navigation
st.title("DataWisata: Sistem Rekomendasi Wisata")
tabs = st.tabs(["Rekomendasi Wisata", "Daftar Wisata", "Rekomendasi Berdasarkan User"])

# Calculate similarities
item_similarity = calculate_item_similarity(rating_df)
tfidf_similarity = get_tfidf_similarity(wisata_df)

# Tab 1: Rekomendasi Wisata
with tabs[0]:
    st.header("Rekomendasi Wisata")

    # Dropdown for wisata selection
    col1, col2, col3 = st.columns(3)

    selected_wisata_1 = col1.selectbox("Wisata 1", ["Pilih Wisata"] + list(wisata_df['wisata']), key="wisata1")
    selected_wisata_2 = col2.selectbox("Wisata 2", ["Pilih Wisata"] + list(wisata_df['wisata']), key="wisata2")

    selected_wisata = [wisata for wisata in [selected_wisata_1, selected_wisata_2] if wisata != "Pilih Wisata"]

    if st.button("Dapatkan Rekomendasi"):
        if not selected_wisata:
            st.warning("Silakan pilih minimal satu tempat wisata.")
        else:
            combined_recommendations = {}
            for wisata in selected_wisata:
                try:
                    recommendations = recommend_wisata(item_similarity, tfidf_similarity, wisata)
                    for rec_wisata, score in recommendations.items():
                        if rec_wisata in combined_recommendations:
                            combined_recommendations[rec_wisata] += score
                        else:
                            combined_recommendations[rec_wisata] = score
                except ValueError as e:
                    st.error(e)

            # Sort and display recommendations
            sorted_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)
            st.subheader("Rekomendasi untuk Anda:")
            for wisata, score in sorted_recommendations[:5]:
                st.write(f"- {wisata} (Score: {score:.2f})")

# Tab 2: Daftar Wisata
with tabs[1]:
    st.header("Daftar Wisata")
    st.dataframe(wisata_df)
    st.subheader("TF-IDF Similarity")
    sns.heatmap(tfidf_similarity, cmap="Blues", xticklabels=tfidf_similarity.columns, yticklabels=tfidf_similarity.columns)
    st.pyplot(plt)
    
    st.header("Daftar Rating Score")
    st.dataframe(rating_df)
    st.subheader("Item-Based Collaborative Filtering")
    sns.heatmap(item_similarity, cmap="Blues", xticklabels=item_similarity.columns, yticklabels=item_similarity.columns)
    st.pyplot(plt)


with tabs[2]:
    st.header("Rekomendasi Berdasarkan User")

    # Input data user profile
    nama_user = st.text_input("Nama Anda")
    jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-Laki", "Perempuan"])

    # Pilih preferensi fasilitas wisata
    st.subheader("Pilih Fasilitas yang Anda Inginkan")
    fasilitas_list = wisata_df['fasilitas'].dropna().str.split(',').explode().unique()
    fasilitas_dipilih = st.multiselect("Fasilitas yang Anda inginkan:", fasilitas_list)

    if st.button("Dapatkan Rekomendasi", key="button_rekomendasi"):
        if not nama_user or not fasilitas_dipilih:
            st.warning("Harap isi nama Anda dan pilih minimal satu fasilitas.")
        else:
            st.write(f"Halo {nama_user}, berikut adalah rekomendasi wisata untuk Anda:")

            # Filter data wisata berdasarkan fasilitas yang dipilih
            def filter_by_fasilitas(wisata_df, fasilitas_dipilih):
                def has_all_fasilitas(row):
                    row_fasilitas = row['fasilitas'].split(',') if pd.notnull(row['fasilitas']) else []
                    return all(f.strip().lower() in [f.strip().lower() for f in row_fasilitas] for f in fasilitas_dipilih)
                return wisata_df[wisata_df.apply(has_all_fasilitas, axis=1)]

            filtered_wisata = filter_by_fasilitas(wisata_df, fasilitas_dipilih)

            if not filtered_wisata.empty:
                # Menghitung rata-rata rating untuk wisata yang terfilter
                filtered_wisata = filtered_wisata.merge(
                    wisata_df.groupby('wisata')['rating'].mean().rename("average_rating"),
                    on='wisata',
                    how='left'
                )
                filtered_wisata['average_rating'] = filtered_wisata['average_rating'].fillna(0)

                # Menghitung TF-IDF similarity untuk wisata yang terfilter
                tfidf_similarity_filtered = get_tfidf_similarity(filtered_wisata)

                # Menggabungkan skor
                final_scores = (
                    tfidf_similarity_filtered.mean(axis=1) * 0.5 +
                    filtered_wisata.set_index('wisata')['average_rating'] * 0.5
                )

                # Menampilkan rekomendasi
                top_recommendations = final_scores.sort_values(ascending=False).head(5)
                for wisata, score in top_recommendations.items():
                    st.write(f"- {wisata}: Skor {score:.2f}")
            else:
                st.write("Tidak ada wisata yang sesuai dengan kriteria.")


