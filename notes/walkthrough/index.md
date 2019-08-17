# Walkthrough

```python
import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
%matplotlib inline
```

## Data collection

```python
url = "https://www.amazon.com/s/ref=sr_nr_p_72_3?fst=as%3Aoff&rh=n%3A165793011%2Cn%3A%21165795011%2Cn%3A196601011%2Cp_n_age_range%3A165813011%2Cp_72%3A1248966011&bbn=196601011&ie=UTF8&qid=1524683296&rnid=1248961011"
```

```python
response = requests.get(url)
```

```python
headers = {
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'
}
response = requests.get(url, headers=headers)
```

```python
root = BeautifulSoup(response.text, "html5lib")
```

```python
items = root.find_all("div", class_="s-item-container")
```

```python
print(items[1].find("img")["src"])
print(items[1].find("a", class_="s-access-detail-page").text)
print(items[1].find("span", class_="sx-price-whole").text)
print(items[1].find("sup", class_="sx-price-fractional").text)
print(items[1].find("i", class_="a-icon-star").text)
print(items[1].find("a", {"href":lambda x : x.endswith("#customerReviews")}).text)

```

```python
response2 = requests.get(items[1].find("img")["src"], headers=headers)
display.Image(data=response2.content)
```

```output
<IPython.core.display.Image object>
```

```python
[x.find("a")["href"] for x in root.find_all("span", class_="pagnLink")][0]
```

```output
'/s/ref=sr_pg_2/141-4642046-9099758?fst=as%3Aoff&rh=n%3A165793011%2Cn%3A%21165795011%2Cn%3A196601011%2Cp_n_age_range%3A165813011%2Cp_72%3A1248966011&page=2&bbn=196601011&ie=UTF8&qid=1525113995'
```

```python
import pickle
```

```python
# ....
urls = []
all_items = []
with open("output.pkl", "rb") as f:
    while True:
        try:
            urls.append(pickle.load(f, encoding='bytes'))
            all_items += pickle.load(f, encoding='bytes')
        except:
            break
all_items = [{k.decode("utf-8"):v for k,v in item.items()} for item in all_items]        
df = pd.DataFrame(all_items)
```

```python
df.iloc[0]
```

```output
img_data       b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...
img_link       https://images-na.ssl-images-amazon.com/images...
link           https://www.amazon.com/VTech-Stand-Learning-Fr...
name           VTech Sit-to-Stand Learning Walker (Frustratio...
num_ratings                                                7,665
price                                                      24.88
stars                                         4.7 out of 5 stars
Name: 0, dtype: object
```

## Visualization and cleaning

```python
df["num_ratings"] = df["num_ratings"].apply(lambda x : float(x.replace(",","")))
df["stars"] = df["stars"].apply(lambda x : float(x.split()[0]))
df["rating"] = (df["stars"]-1)/4
```

```python
plt.hist(np.log10(df["num_ratings"]), bins=25)
```

![](output_0.png)

```python
plt.hist(df["rating"], bins=25)
```

![](output_1.png)

```python
plt.scatter(np.log10(df["num_ratings"]), df["rating"])
```

![](output_2.png)

```python
df["bayes_rating"] = (df["rating"]*df["num_ratings"] + 1)/(df["num_ratings"]+2)
plt.scatter(np.log10(df["num_ratings"]), df["bayes_rating"])
```

![](output_3.png)

```python
plt.hist(np.log10(df["price"][~pd.isnull(df["price"])]), bins=25)
```

![](output_4.png)

```python
df.dropna(subset=["price"], inplace=True)
```

```python
plt.scatter(np.log10(df["price"]), df["bayes_rating"])
```

![](output_5.png)

```python
df.sort_values(by=["bayes_rating"], ascending=False, inplace=True)
```

```python
df = df[df["num_ratings"]>4].copy()
```

## First (rarely successful) take on machine learning

```python
plt.hist(df["bayes_rating"], bins=25)
```

![](output_6.png)

```python
import scipy.stats as stats
```

```python
stats.probplot((df["bayes_rating"] - np.mean(df["bayes_rating"]))/np.std(df["bayes_rating"]),plot=plt)
```

![](output_7.png)

```python
df["logit_rating"] = np.log(df["bayes_rating"]/(1-df["bayes_rating"]))
```

```python
plt.hist(df["logit_rating"], bins=25)
```

![](output_8.png)

```python
stats.probplot((df["logit_rating"] - np.mean(df["logit_rating"]))/np.std(df["logit_rating"]),plot=plt)
```

![](output_9.png)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=10, max_df=0.3, ngram_range=(1,3))


X_tfidf = tfidf.fit_transform(df["name"]).toarray()
X = np.hstack([X_tfidf, np.log(df[["price"]]).values])
y = df["logit_rating"].values
```

```python
from sklearn.model_selection import train_test_split
X_, X_test, y_, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
X_train, X_cv, y_train, y_cv = train_test_split(X_, y_, test_size = 0.3, random_state=0)
```

```python
from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)
```

```output
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
```

```python
print("Training MSE: ", np.mean((clf.predict(X_train) - y_train)**2))
print("Baseline training MSE: ", np.mean((np.mean(y_train) - y_train)**2))
```

```python
print("CV MSE: ", np.mean((clf.predict(X_cv) - y_cv)**2))
print("Baseline CV MSE: ", np.mean((np.mean(y_train) - y_cv)**2))
```

```python
print("Test MSE: ", np.mean((clf.predict(X_test) - y_test)**2))
print("Baseline Test MSE: ", np.mean((np.mean(y_train) - y_test)**2))
```

## Playing a game

```python
def display_item(s):
    display.display(display.Image(data = s["img_data"]))
    print(s["name"])
    print("$" + str(np.round(s["price"],3)))

```

```python
def ranking_game():
    it1 = np.random.randint(len(df))
    it2 = np.random.randint(len(df))
    
    display_item(df.iloc[it1])
    print("-------")
    display_item(df.iloc[it2])
    print("-------")
    
    guess = int(input("Which item is rated higher (1/2)?"))
    
    print("True Rating: {} / {}".format(df.iloc[it1]["bayes_rating"], df.iloc[it2]["bayes_rating"]))
    correct = ((guess == 1) and (it1 < it2)) or ((guess == 2) and (it1 > it2))
    print("{}".format("Correct!" if correct else "Incorrect!"))
```

```python
ranking_game()
```

```output
<IPython.core.display.Image object>
```

```python
def ranking_game2():
    it1 = np.random.randint(len(df)//10)
    it2 = np.random.randint(len(df)//10) + 9*len(df)//10
    if np.random.rand() < 0.5:
        it1,it2 = (it2,it1)
    
    display_item(df.iloc[it1])
    print("-------")
    display_item(df.iloc[it2])
    print("-------")
    
    guess = int(input("Which item is rated higher (1/2)?"))
    
    print("True Rating: {} / {}".format(df.iloc[it1]["bayes_rating"], df.iloc[it2]["bayes_rating"]))
    correct = ((guess == 1) and (it1 < it2)) or ((guess == 2) and (it1 > it2))
    print("{}".format("Correct!" if correct else "Incorrect!"))
```

```python
ranking_game2()
```

```output
<IPython.core.display.Image object>
```

## Classification

```python
idx = ((df["bayes_rating"] > np.percentile(df["bayes_rating"], 90)) |
       (df["bayes_rating"] < np.percentile(df["bayes_rating"], 10))).values
y = df["bayes_rating"] > np.median(df["bayes_rating"])
```

```python
X_, X_test, y_, y_test = train_test_split(X[idx,:], y[idx], test_size = 0.3, random_state=0)
X_train, X_cv, y_train, y_cv = train_test_split(X_, y_, test_size = 0.3, random_state=0)
```

```python
X_train.shape
```

```output
(277, 669)
```

```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_cv, y_cv))
```

```python
def get_err_samples(clf, X, y, X_cv, y_cv, n_samples):
    clf.fit(X[:n_samples], y[:n_samples])
    return np.array([clf.score(X[:n_samples], y[:n_samples]), clf.score(X_cv, y_cv)])

n_samples = [int(x) for x in np.linspace(20, X_train.shape[0], 10)]
errors = np.array([get_err_samples(clf, X_train, y_train, X_cv, y_cv, n) for n in n_samples])
plt.plot(n_samples, errors[:,0], n_samples, errors[:,1])
plt.legend(["Training", "Validation"])
```

![](output_10.png)

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
errors = np.array([get_err_samples(clf, X_train, y_train, X_cv, y_cv, n) for n in n_samples])
plt.plot(n_samples, errors[:,0], n_samples, errors[:,1])
plt.legend(["Training", "Validation"])
```

![](output_11.png)

```python
def get_err_C(clf, X, y, X_cv, y_cv, C):
    clf.C = C
    clf.fit(X, y)
    return np.array([clf.score(X, y), clf.score(X_cv, y_cv)])

C = np.logspace(-4, 4, 50)
errors = np.array([get_err_C(clf, X_train, y_train, X_cv, y_cv, c) for c in C])
plt.semilogx(C, errors[:,0], C, errors[:,1])
plt.legend(["Training", "Validation"])
```

![](output_12.png)

```python
clf = LogisticRegression(C=1.0)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_cv, y_cv))
print(clf.score(X_test, y_test))
```

```python
feature_names = np.array(tfidf.get_feature_names() + ["++price++"])
```

```python
sorted(zip(feature_names, clf.coef_[0]), key=lambda x : np.abs(x[1]), reverse=True)[:20]
```

```output
[('bath', -1.0998582551786644),
 ('and', 1.0954028039409269),
 ('green toys', 0.98395273227295454),
 ('frog', -0.91726295727875706),
 ('go', 0.87272474343975692),
 ('imaginext', 0.78991692736571695),
 ('doug', 0.78020112442135148),
 ('melissa doug', 0.78020112442135148),
 ('melissa', 0.77905357055216928),
 ('fisher', 0.76947656750321813),
 ('fisher price', 0.76947656750321813),
 ('price', 0.76947656750321813),
 ('with', 0.74108288761147312),
 ('playskool', 0.7270449833570779),
 ('crayola', -0.71886844630886249),
 ('kids', -0.69337146109167946),
 ('little tikes', -0.67831342789823301),
 ('tikes', -0.67831342789823301),
 ('piece', 0.67212093384960425),
 ('puzzle', 0.66734431522410265)]
```

## Image features

```python
with open("image_features.npy", "rb") as f:
    X_img = np.load(f)
X = X_img
```

```python
X_, X_test, y_, y_test = train_test_split(X[idx,:], y[idx], test_size = 0.3, random_state=0)
X_train, X_cv, y_train, y_cv = train_test_split(X_, y_, test_size = 0.3, random_state=0)
```

```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_cv, y_cv))
```

```python
C = np.logspace(-10, 10, 50)
errors = np.array([get_err_C(clf, X_train, y_train, X_cv, y_cv, c) for c in C])
plt.semilogx(C, errors[:,0], C, errors[:,1])
plt.legend(["Training", "Validation"])
```

![](output_13.png)

```python
clf = LogisticRegression(C=1e-4)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_cv, y_cv))
print(clf.score(X_test, y_test))
```

## Images and text

```python
print(np.linalg.norm(X_tfidf))
print(np.linalg.norm(X_img))
```

```python
X = np.hstack([X_tfidf, X_img/100, np.log(df[["price"]].values)])
X_, X_test, y_, y_test = train_test_split(X[idx,:], y[idx], test_size = 0.3, random_state=0)
X_train, X_cv, y_train, y_cv = train_test_split(X_, y_, test_size = 0.3, random_state=0)
```

```python
clf = LogisticRegression()
C = np.logspace(-4, 4, 50)
errors = np.array([get_err_C(clf, X_train, y_train, X_cv, y_cv, c) for c in C])
plt.semilogx(C, errors[:,0], C, errors[:,1])
plt.legend(["Training", "Validation"])
```

![](output_14.png)

```python
clf = LogisticRegression(C=0.5)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_cv, y_cv))
print(clf.score(X_test, y_test))
```

## Finding the best underreviewed toys

```python
# ....
urls = []
all_items = []
with open("output.pkl", "rb") as f:
    while True:
        try:
            urls.append(pickle.load(f, encoding='bytes'))
            all_items += pickle.load(f, encoding='bytes')
        except:
            break
all_items = [{k.decode("utf-8"):v for k,v in item.items()} for item in all_items]        
df0 = pd.DataFrame(all_items)


df0["num_ratings"] = df0["num_ratings"].apply(lambda x : float(x.replace(",","")))
df0["stars"] = df0["stars"].apply(lambda x : float(x.split()[0]))
df0["rating"] = (df0["stars"]-1)/4
```

```python
df_deploy = df0[df0["num_ratings"] <=4].copy()
df_deploy.dropna(subset=["price"], inplace=True)
```

```python
X_deploy = np.hstack([tfidf.transform(df_deploy.name).toarray(), np.log(df_deploy[["price"]].values)])
```

```python
deploy_scores = clf.decision_function(X_deploy)
df_deploy["score"] = deploy_scores
```

```python
df_deploy.sort_values(by=["score"], ascending=False).iloc[:20]["name"]
```

```output
484       Fisher-Price Imaginext Power Ranger Battle Pack
219     Fisher-Price Imaginext DC Super Friends Batman...
1538    Fisher-Price Imaginext Monsters University Sca...
3546                  Fisher-Price Imaginext Warrior Suit
5151           Fisher Price Record Player with 3 records.
3205    Fisher-Price Imaginext Davey Jones & Triple Ca...
2147    Fisher-Price Imaginext Power Rangers Battle Ar...
140     Fisher-Price Imaginext DC Super Friends Lex Co...
3491            Fisher-Price Little People Wheelies Eddie
5199    Fisher-Price Little People Wheelies Recycle Truck
2763                  Fisher-Price Little People Flamingo
345               Fisher-Price Little People Photographer
3563    Fisher-Price Little People Wheelies Ice Cream ...
156                    Fisher-Price Little People Ostrich
2991              Fisher-Price Little People Veterinarian
290     Fisher-Price Imaginext Dc Super Friends Figure...
1539                   Fisher-Price Little People Teacher
1253                Fisher-Price Little People Police Car
331                     Fisher-Price Little People Artist
2779                     Fisher-Price Little People Hippo
Name: name, dtype: object
```
