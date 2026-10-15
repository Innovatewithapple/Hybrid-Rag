
overveiw = Overview_quesiton_Embeddings(embeddings=embeddings)
incoming = Incoming_quesiton_Embeddings(embeddings=embeddings,question='What are the main objectives of the GDPR?')

overveiw = np.array(overveiw)
incoming = np.array(incoming)
print(overveiw.shape)
print(incoming.shape)
scores = overveiw @ incoming
best_index = scores.argmax()

print("Best match:", overview_questions[best_index])
print("Similarity:", scores[best_index])