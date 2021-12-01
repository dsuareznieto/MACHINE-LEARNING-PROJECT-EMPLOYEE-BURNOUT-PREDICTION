
def label_encoder(data,column):
    '''Esta función realiza label encoding'''
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(data[column])
    data[column]=le.transform(data[column])