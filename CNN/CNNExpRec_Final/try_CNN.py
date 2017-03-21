import fer_parser
import csv

parser = fer_parser.Fer_Parser()
X_train, Y_train, X_test, Y_test = parser.parse_all()
flipped_X_train = X_train[:,:,::-1]

print X_train
print '-----------------------------'
print flipped_X_train

#csvfile = file('X_train.csv', 'wb')
#writer = csv.writer(csvfile)
#writer.writerows(X_train)
#csvfile.close()

#csvfile2 = file('flipped_X_train.csv', 'wb')
#writer2 = csv.writer(csvfile2)
#writer2.writerows(flipped_X_train)
#csvfile2.close()