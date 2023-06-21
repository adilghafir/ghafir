import streamlit as st
import pandas as pd
import pickle
from sklearn import svm, neighbors, tree
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

st.sidebar.write("MASTER DSEF")
st.sidebar.write("Prepared by: ADIL GHAFIR")
st.sidebar.write("Supervised by: MR TALI")
# Charger les données depuis le fichier CSV
df = pd.read_csv('prices.csv')
y = df['Value']
X = df[['Rooms', 'Distance']]

# Entraîner les modèles
svm_model = svm.SVR()
svm_model.fit(X, y)

knn_model = neighbors.KNeighborsRegressor()
knn_model.fit(X, y)

tree_model = tree.DecisionTreeRegressor()
tree_model.fit(X, y)

pca_model = PCA()
pca_model.fit(X)

linear_model = LinearRegression()
linear_model.fit(X, y)

# Fonction pour effectuer une prédiction
def predict_price(model, rooms, distance):
    prediction = model.predict([[rooms, distance]])
    return prediction

# Interface utilisateur avec Streamlit
def main():
    st.title("Prédiction des prix")
    
    # Saisie des variables indépendantes
    rooms = st.number_input("Nombre de chambres", value=0, step=1)
    distance = st.number_input("Distance", value=0.0, step=0.1)
    
    # Bouton pour effectuer la prédiction
    if st.button("Prédire"):
        svm_prediction = predict_price(svm_model, rooms, distance)
        knn_prediction = predict_price(knn_model, rooms, distance)
        tree_prediction = predict_price(tree_model, rooms, distance)
        linear_prediction = predict_price(linear_model, rooms, distance)
        
        st.success("Prédictions :")
        st.write("SVM : ", svm_prediction)
        st.write("KNN : ", knn_prediction)
        st.write("Decision Tree : ", tree_prediction)
        st.write("Linear Regression : ", linear_prediction)
    
    # Affichage du score des modèles
    st.subheader("Score des modèles")
    st.write("SVM : ", svm_model.score(X, y))
    st.write("KNN : ", knn_model.score(X, y))
    st.write("Decision Tree : ", tree_model.score(X, y))
    st.write("Linear Regression : ", linear_model.score(X, y))

if __name__ == "__main__":
    main()
