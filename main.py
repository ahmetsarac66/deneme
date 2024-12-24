# https://www.kaggle.com/code/franciscosantos2/is-99-accuracy-good-maybe-not-credit-card-fraud/notebook
# https://www.kaggle.com/code/nathanxiang/credit-card-fraud-analysis-and-modeling/notebook

#########################################################################################################################################################

lib_path = "/libs"
data_path = "/datas"

try:
    # Colabda çalışıyor ise ona göre ayarlamalar yapıyoruz
    project_path = '/content/drive/My Drive/BitirmeOdevi'

    # Google Driverden veri alabilmek için, google drive'yi colaba bağlıyoruz
    from google.colab import drive

    drive.mount('/content/drive')

    # kütüphanelerimiz ve data klasörlerimizi daha rahat kullanabilmek için kısayollar linklemelerini değiştiriyoruz

    from os import symlink
    from sys import path

    # kütüphaneler için
    try:
        symlink(project_path + lib_path, lib_path)
        # eğer zaten oluşturuluşsa pass geçiyoruz
    except FileExistsError:
        pass
    finally:
        path.insert(0, lib_path)

    # datalar için
    try:
        symlink(project_path + data_path, data_path)
    except FileExistsError:
        # eğer zaten oluşturuluşsa pass geçiyoruz
        pass
    finally:
        path.insert(0, data_path)

    # proje klasörünü de ekliyoruz
    path.insert(0, project_path)

    # gerekli kütüphaneleri yüklüyoruz, bu işlem uzun sürebilir

    # !pip install -r "/content/drive/My Drive/BitirmeOdevi/requirements.txt"

    IN_COLAB = True
except ImportError:
    # eğer colabda değil ise, absolute pathları relativeye çeviriyoruz
    project_path = '.'
    lib_path = project_path + lib_path
    data_path = project_path + data_path
    IN_COLAB = False

#########################################################################################################################################################

from pandas import read_csv, set_option

from libs.data_process import organize_data

# CSV leri içe aktarıyoruz, index verisini csvdeki ilk indexte bulunan veriyi baz alıyoruz
test_dataframe = read_csv(data_path + "/fraudTest.csv", index_col=0)
train_dataframe = read_csv(data_path + "/fraudTrain.csv", index_col=0)

# Verilerin miktarına bakıyoruz
train_shape = train_dataframe.shape
test_shape = test_dataframe.shape
total_shape = train_shape[0] + test_shape[0]

print("Before organizing")
print("Train: {} rows and {} columns".format(*train_shape))
print("Test: {} rows and {} columns".format(*test_shape))
print(f"Ratio: test/total, %{(test_shape[0] / total_shape) * 100:.2f}")
print()

# verilerimizi organize ediyoruz
test_dataframe = organize_data(test_dataframe)
train_dataframe = organize_data(train_dataframe)

# Verilerin miktarına bakıyoruz
train_shape = train_dataframe.shape
test_shape = test_dataframe.shape
total_shape = train_shape[0] + test_shape[0]
print("After organizing")
print("Train: {} rows and {} columns".format(*train_shape))
print("Test: {} rows and {} columns".format(*test_shape))
print(f"Ratio: test/total, %{(test_shape[0] / total_shape) * 100:.2f}")
print()

# verilerin son halini print ediyoruz
set_option('display.max_columns', None)
print(train_dataframe.head(5))

# Öğrenim verilerini, değerler ve sonuç olarak ayırıyoruz
X_train = train_dataframe.drop(columns=["is_fraud"])
Y_train = train_dataframe["is_fraud"]

# aynısını test verileri için de yapıyoruz

X_test = test_dataframe.drop(columns=["is_fraud"])
Y_test = test_dataframe["is_fraud"]

# Train verilerini, train ve valid olarak ayıklıyoruz
from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.2)

#########################################################################################################################################################

# modellerimizi oluşturuyoruz, doğrulu en yüksek olanı kullanacağız


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

models = {
    "rf": {
        "model": RandomForestClassifier(random_state=23)
    },
    "knn": {
        "model": KNeighborsClassifier()
    },
    "gboost": {
        "model": GradientBoostingClassifier(random_state=23)
    },
    "lgbm": {
        "model": LGBMClassifier(random_state=23)
    }
}

from libs.model_process import evaluate_model

for _modelName, _modelData in models.items():
    print("Training", _modelName)
    print()
    _model = _modelData["model"]
    _model.fit(X_train, Y_train)

    # modelimizi test ediyoruz
    y_pred_train = _model.predict(X_train)
    y_score_train = _model.predict_proba(X_train)[:, 1]

    y_pred_test = _model.predict(X_test)
    y_score_test = _model.predict_proba(X_test)[:, 1]

    # train metrics
    print("Train Metrics")
    evaluate_model(Y_train, y_pred_train, y_score_train)
    print()

    # test metrics
    print("Test Metrics")
    accuracy, precision, recall, f1, auc = evaluate_model(Y_test, y_pred_test, y_score_test)
    print()
    _modelData["accuracy"] = accuracy
    _modelData["precision"] = precision
    _modelData["recall"] = recall
    _modelData["f1"] = f1
    _modelData["auc"] = auc

# öğrenim detaylarını print ettiriyoruz
print(models)

# doğruluğu en yüksek olan modeli alıyoruz
_max = max([x["accuracy"] for x in models.values()])
modelData = {x: y for x, y in models.items() if y["accuracy"] == _max}
print(modelData)

# modelimizi kayıt ediyoruz
import joblib
joblib.dump(modelData[list(modelData.keys())[0]]["model"], project_path + "/ai.model")
