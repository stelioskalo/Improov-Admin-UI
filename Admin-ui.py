import tkinter
import requests
from bs4 import BeautifulSoup
from firebase import firebase
import pickle
import pandas as pd
import nltk
from random import randrange
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')

firebase = firebase.FirebaseApplication("https://improov-6972e.firebaseio.com/")

path_knn = 'best_knnc.pickle'
with open(path_knn, 'rb') as data:
    knn_model = pickle.load(data)

path_tfidf = "tfidf.pickle"
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

category_codes = {
    'Advertising\n': 0,
    'Entrepreneurship\n': 1,
    'Accounting\n': 2,
    'Audit\n': 3,
    'Banking\n': 4,
    'Corporate Law\n': 5,
    'Finance\n': 6,
    'Ecommerce\n': 7,
    'Ethics\n': 8,
    'Human Resource\n': 9,
    'Insurance\n': 10,
    'Investing\n': 11,
    'Logistics\n': 12,
    'Marketing\n': 13,
    'Negotiation\n': 14,
    'Real Estate\n': 15,
    'Sales\n': 16,
    'Startup\n': 17,
    'Technology\n': 18,
    'Trading\n': 19,
    'Writing\n': 20
}

punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))


def create_features_from_text(text):
    # Dataframe creation
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['Article'])
    df.loc[0] = text
    df['Article_Parsed_1'] = df['Article'].str.replace("\r", " ")
    df['Article_Parsed_1'] = df['Article_Parsed_1'].str.replace("\n", " ")
    df['Article_Parsed_1'] = df['Article_Parsed_1'].str.replace("    ", " ")
    df['Article_Parsed_1'] = df['Article_Parsed_1'].str.replace('"', '')
    df['Article_Parsed_2'] = df['Article_Parsed_1'].str.lower()
    df['Article_Parsed_3'] = df['Article_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Article_Parsed_3'] = df['Article_Parsed_3'].str.replace(punct_sign, '')
    df['Article_Parsed_4'] = df['Article_Parsed_3'].str.replace("'s", "")
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text = df.loc[0]['Article_Parsed_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)
    lemmatized_text_list.append(lemmatized_text)
    df['Article_Parsed_5'] = lemmatized_text_list
    df['Article_Parsed_6'] = df['Article_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Article_Parsed_6'] = df['Article_Parsed_6'].str.replace(regex_stopword, '')
    df = df['Article_Parsed_6']

    # TF-IDF
    features = tfidf.transform(df).toarray()

    return features


def get_category_name(category_id):
    for category, id_ in category_codes.items():
        if id_ == category_id:
            return category


def predict_from_text(text):
    # Predict using the input model
    prediction_knn = knn_model.predict(create_features_from_text(text))[0]
    prediction_knn_proba = knn_model.predict_proba(create_features_from_text(text))[0]

    # Return result
    category_knn = get_category_name(prediction_knn)

    print("The predicted category using the lrc model is %s." % (category_knn))
    print("The conditional probability is: %a" % (prediction_knn_proba.max() * 100))
    return category_knn.rstrip()


window = tkinter.Tk()


def create_window():
    window2 = tkinter.Toplevel(window)
    window2.title("Add Article")
    tkinter.Label(window2, text="").pack()
    tkinter.Label(window2, text="").pack()
    tkinter.Label(window2, text="Add Article by entering its URL").pack()
    window2.geometry('500x200')
    tkinter.Label(window2, text="").pack()
    entry1 = tkinter.Entry(window2, width=50)
    entry1.pack()
    entry1.insert(0, "Enter a URL")
    tkinter.Label(window2, text="").pack()

    def add_article():

        if entry1.get().startswith('https://www.entrepreneur.com/article/'):
            entries = firebase.get('/article/', '')
            values = list(entries.values())
            print(values[0].get('url'))
            match = False
            for i in range(len(values)):
                if values[i].get('url') == entry1.get():
                    match = True

            if match == False:

                r1 = requests.get(entry1.get())
                coverpage = r1.content
                soup1 = BeautifulSoup(coverpage, 'html.parser')
                paragraphs = ["" for x in range(40)]
                paragraphs_retrieved = [None] * 40
                paragraphs_retrieved = soup1.find_all('p')
                for i in range(len(paragraphs_retrieved)):
                    if paragraphs_retrieved[i] is None:
                        paragraphs[i] = ""
                    else:
                        paragraphs[i] = paragraphs_retrieved[i].getText()

                heading = soup1.find('h1')
                title = heading.getText().strip()
                full_article = ""

                for i in range(len(paragraphs_retrieved) - 2):
                    full_article = full_article + paragraphs_retrieved[i].getText() + " "

                randomnum = randrange(5) + 1

                category = str(predict_from_text(full_article))
                upload = {
                    'title': title,
                    'url': entry1.get(),
                    'category': category,
                    'imagenum': str(randomnum),
                    'paragraph1': paragraphs[0],
                    'paragraph2': paragraphs[1],
                    'paragraph3': paragraphs[2],
                    'paragraph4': paragraphs[3],
                    'paragraph5': paragraphs[4],
                    'paragraph6': paragraphs[5],
                    'paragraph7': paragraphs[6],
                    'paragraph8': paragraphs[7],
                    'paragraph9': paragraphs[8],
                    'paragraph10': paragraphs[9],
                    'paragraph11': paragraphs[10],
                    'paragraph12': paragraphs[11],
                    'paragraph13': paragraphs[12],
                    'paragraph14': paragraphs[13],
                    'paragraph15': paragraphs[14],
                    'paragraph16': paragraphs[15],
                    'paragraph17': paragraphs[16],
                    'paragraph18': paragraphs[17],
                    'paragraph19': paragraphs[18],
                    'paragraph20': paragraphs[19],
                    'paragraph21': paragraphs[20],
                    'paragraph22': paragraphs[21],
                    'paragraph23': paragraphs[22],
                    'paragraph24': paragraphs[23],
                    'paragraph25': paragraphs[24],
                    'paragraph26': paragraphs[25],
                    'paragraph27': paragraphs[26],
                    'paragraph28': paragraphs[27],
                    'paragraph29': paragraphs[28],
                    'paragraph30': paragraphs[29],
                    'paragraph31': paragraphs[30],
                    'paragraph32': paragraphs[31],
                    'paragraph33': paragraphs[32],
                    'paragraph34': paragraphs[33],
                    'paragraph35': paragraphs[34],
                    'paragraph36': paragraphs[35],
                    'paragraph37': paragraphs[36],
                    'paragraph38': paragraphs[37],
                    'paragraph39': paragraphs[38],
                    'paragraph40': paragraphs[39],
                }
                result = firebase.post('/article/', upload)
                tkinter.Label(window2, text="Article Added!").pack()
            else:
                tkinter.Label(window2, text="That article is already in the database, add a new one!").pack()
        else:
            tkinter.Label(window2, text="Please enter a valid URL").pack()

        tkinter.Label(window2, text="").pack()
        tkinter.Label(window2, text=category).pack()
        tkinter.Label(window2, text="").pack()
        tkinter.Label(window2, text=title).pack()

    tkinter.Button(window2, text="Add article", width=20, command=add_article).pack()


def create_window2():
    window3 = tkinter.Toplevel(window)
    window3.title("Create Discussion")
    tkinter.Label(window3, text="").pack()
    tkinter.Label(window3, text="").pack()
    tkinter.Label(window3, text="Add what the discussion is about").pack()
    tkinter.Label(window3, text="").pack()
    entry1 = tkinter.Entry(window3, width=70)
    entry1.pack()
    entry1.insert(0, "Enter discussion description")
    window3.geometry('500x350')
    tkinter.Label(window3, text="").pack()
    tkinter.Label(window3, text="Select topic of discussion").pack()
    tkinter.Label(window3, text="").pack()
    topic = tkinter.StringVar()
    topic.set("Advertising")
    list = tkinter.OptionMenu(window3, topic, "Advertising",
                              "Accounting",
                              "Entrepreneurship",
                              "Audit",
                              "Banking",
                              "Corporate Law",
                              "Finance",
                              "Ecommerce",
                              "Ethics",
                              "Human Resource",
                              "Insurance",
                              "Investing",
                              "Logistics",
                              "Marketing",
                              "Negotiation",
                              "Real Estate",
                              "Sales",
                              "Startup",
                              "Technology",
                              "Trading",
                              "Writing").pack()
    tkinter.Label(window3, text="").pack()
    tkinter.Label(window3, text="Write the first post").pack()
    tkinter.Label(window3, text="").pack()
    entry2 = tkinter.Entry(window3, width=70)
    entry2.pack()
    entry2.insert(0, "First post")

    def add_discussion():
        if entry1.get() == "" or entry2.get() == "":
            tkinter.Label(window3, text="Please enter a valid post and description").pack()
        else:
            upload = {
                'about': entry1.get(),
                'creatorname': 'Improov',
                'topic': topic.get(),
                'creatorid': '5CR34CR19Sa1Aw0YfeyQMl16dLG3',
                'post1': {
                    'post': entry2.get(),
                    'postuser': '5CR34CR19Sa1Aw0YfeyQMl16dLG3'
                }
            }
            firebase.post('/discussion/', upload)
            tkinter.Label(window3, text="Discussion Added!").pack()

    tkinter.Label(window3, text="").pack()
    tkinter.Button(window3, text="Add Discussion", width=20, command=add_discussion).pack()

def create_window3():
    window4 = tkinter.Toplevel(window)
    window4.title("View User Reports")
    tkinter.Label(window4, text="").grid(row=1, column=0)
    tkinter.Label(window4, text="").grid(row=2, column=0)
    tkinter.Label(window4, text="From", font='Helvetica 10 bold').grid(row=3, column=0)
    tkinter.Label(window4, text="                      ").grid(row=3, column=1)
    tkinter.Label(window4, text="To", font='Helvetica 10 bold').grid(row=3, column=2)
    tkinter.Label(window4, text="                               ").grid(row=3, column=3)
    tkinter.Label(window4, text="Message", font='Helvetica 10 bold').grid(row=3, column=4)
    tkinter.Label(window4, text="").grid(row=3, column=5)
    tkinter.Label(window4, text="                                             ").grid(row=3, column=6)
    tkinter.Label(window4, text="Date", font='Helvetica 10 bold').grid(row=3, column=7)
    tkinter.Label(window4,text="           ").grid(row=3, column=8)
    tkinter.Label(window4, text="").grid(row=4, column=0)
    reports = firebase.get('/report/', '')
    values = list(reports.values())
    for i in range(len(values)):
        tkinter.Label(window4, text=values[i].get('from')).grid(row=4 + i + 1, column=0)
        tkinter.Label(window4, text=values[i].get('to')).grid(row=4 + i + 1, column=2)
        tkinter.Label(window4, text=values[i].get('message'), wraplength=450).grid(row=4 + i + 1, column=4)
        tkinter.Label(window4, text=values[i].get('date')).grid(row=4 + i + 1, column=7)
    tkinter.Label(window4, text="").grid(row=100, column=0)
    tkinter.Label(window4, text="").grid(row=100, column=0)


window.title("Admin UI")
tkinter.Label(window, text="").pack()
label = tkinter.Label(window, text="Welcome").pack()
tkinter.Label(window, text="").pack()
button = tkinter.Button(window, text="Add article", width=20, command=create_window).pack()
tkinter.Label(window, text="").pack()
tkinter.Button(window, text="Create Discussion", width=20, command=create_window2).pack()
tkinter.Label(window, text="").pack()
tkinter.Button(window, text="View user reports", width=20, command=create_window3).pack()
window.geometry('300x250')
window.mainloop()