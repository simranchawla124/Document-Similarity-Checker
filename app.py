#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:43:40 2021

@author: anonymous
"""

from flask import Flask,render_template,request
from flask import send_file
from flask import after_this_request
app=Flask(__name__)
import gensim
import numpy as np
import os
@app.route('/')
def helloworld():
    return  render_template("similarity.html")
@app.route('/home')
def home():
    return render_template("similarity.html")
@app.route('/about')
def about():
    return  render_template("about.html")

@app.route('/Semantic',methods=['GET','POST'])
def semantic():
     if(request.method=='POST'):
         f=request.files['key']
         f1=request.files['paper']
         filename, file_extension = os.path.splitext(f.filename)
         filename1, file_extension1 = os.path.splitext(f1.filename)
         f.save(os.path.join("uploads",f.filename))
         f1.save(os.path.join("uploads",f1.filename))
         if (file_extension=='.txt'):
             key = open(str(os.path.abspath(str("uploads/"+f.filename))), "r") 
         if (file_extension1=='.txt'):
             paper = open(str(os.path.abspath(str("uploads/"+f1.filename))), "r") 
         if (file_extension=='.pdf'):
             import PyPDF2
             a=PyPDF2.PdfFileReader("/home/anonymous/Desktop/DocSimilarity/Flask/uploads/"+filename+".pdf","rb")
             str1=""
             for i in range(0,a.getNumPages()):
                 str1+=a.getPage(i).extractText()
             with open("uploads/"+filename+".txt","w",encoding='utf-8') as f3:
                
                f3.write(str1)
             key = open("uploads/"+filename+".txt", "r")
            
         if (file_extension1=='.pdf'):
             import PyPDF2
             a=PyPDF2.PdfFileReader("/home/anonymous/Desktop/DocSimilarity/Flask/uploads/"+filename1+".pdf","rb")
             str1=""
             for i in range(0,a.getNumPages()):
                 str1+=a.getPage(i).extractText()
             with open("uploads/"+filename1+".txt","w",encoding='utf-8') as f2:
                
                f2.write(str1)
             paper = open("uploads/"+filename1+".txt", "r") 
         if(file_extension=='.docx'):
             import docx2txt
             str1 = docx2txt.process("uploads/"+str(f.filename))
             with open("uploads/"+filename+".txt","w",encoding='utf-8') as f2:
                
                f2.write(str1)
             key = open("uploads/"+filename+".txt", "r")
         if(file_extension1=='.docx'):
             import docx2txt
             str1 = docx2txt.process("uploads/"+str(f1.filename))
             with open("uploads/"+filename1+".txt","w",encoding='utf-8') as f2:
                
                f2.write(str1)
             paper = open("uploads/"+filename1+".txt", "r")
# Load Google's pre-trained Word2Vec model.
     model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True,limit=300000)
     with paper as file:
         data = file.read().replace('\n', '')
     with key as file2:
         data2 = file2.read().replace('\n', '')
     class DocSim:
         def __init__(self, w2v_model, stopwords=None):
             self.w2v_model = w2v_model
             self.stopwords = stopwords if stopwords is not None else []
         def vectorize(self, doc: str) -> np.ndarray:
             """
             Identify the vector values for each word in the given document
             :param doc:
             :return:
             """
             doc = doc.lower()
             words = [w for w in doc.split(" ") if w not in self.stopwords]
             word_vecs = []
             for word in words:
                 try:
                     vec = self.w2v_model[word]
                     word_vecs.append(vec)
                 except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                     pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
             vector = np.mean(word_vecs, axis=0)
             return vector

         def _cosine_sim(self, vecA, vecB):
             """Find the cosine similarity distance between two vectors."""
             csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
             if np.isnan(np.sum(csim)):
                 return 0
             return csim

         def calculate_similarity(self, source_doc, target_docs=None, threshold=0):
             """Calculates & returns similarity scores between given source document & all
             the target documents."""
             if not target_docs:
                 return []

             if isinstance(target_docs, str):
                 target_docs = [target_docs]

             source_vec = self.vectorize(source_doc)
             results ={}
             for doc in target_docs:
                 target_vec = self.vectorize(doc)
                 sim_score = self._cosine_sim(source_vec, target_vec)
                 if sim_score > threshold:
                     results.update({"score": sim_score})
                     
                 

             return results.get("score")

     ds = DocSim(model)


# In[ ]:


     sim_scores = ds.calculate_similarity(data2,data)
     with open('results.txt','a') as f2:
                         f2.write(filename1)
                         f2.write(":")
                         f2.write(str(sim_scores*100))
                         f2.write('\n')
     return render_template("predict.html",value=sim_scores*100)
@app.route('/Syntax',methods=['GET','POST'])
def syntax():
    if(request.method=='POST'):
         f=request.files['key']
         f1=request.files['paper']
         filename, file_extension = os.path.splitext(f.filename)
         filename1, file_extension1 = os.path.splitext(f1.filename)
         f.save(os.path.join("uploads",f.filename))
         f1.save(os.path.join("uploads",f1.filename))
         if (file_extension=='.txt'):
             key = open(str(os.path.abspath(str("uploads/"+f.filename))), "r") 
         if (file_extension1=='.txt'):
             paper = open(str(os.path.abspath(str("uploads/"+f1.filename))), "r") 
         if (file_extension=='.pdf'):
             import PyPDF2
             a=PyPDF2.PdfFileReader("/home/anonymous/Desktop/DocSimilarity/Flask/uploads/"+filename+".pdf","rb")
             str1=""
             for i in range(0,a.getNumPages()):
                 str1+=a.getPage(i).extractText()
             with open("uploads/"+filename+".txt","w",encoding='utf-8') as f3:
                
                f3.write(str1)
             key = open("uploads/"+filename+".txt", "r")
            
         if (file_extension1=='.pdf'):
             import PyPDF2
             a=PyPDF2.PdfFileReader("/home/anonymous/Desktop/DocSimilarity/Flask/uploads/"+filename1+".pdf","rb")
             str1=""
             for i in range(0,a.getNumPages()):
                 str1+=a.getPage(i).extractText()
             with open("uploads/"+filename1+".txt","w",encoding='utf-8') as f2:
                
                f2.write(str1)
             paper = open("uploads/"+filename1+".txt", "r")
         if(file_extension=='.docx'):
             import docx2txt
             str1 = docx2txt.process("uploads/"+str(f.filename))
             with open("uploads/"+filename+".txt","w",encoding='utf-8') as f2:
                
                f2.write(str1)
             key = open("uploads/"+filename+".txt", "r")
         if(file_extension1=='.docx'):
             import docx2txt
             str1 = docx2txt.process("uploads/"+str(f1.filename))
             with open("uploads/"+filename1+".txt","w",encoding='utf-8') as f2:
                
                f2.write(str1)
             paper = open("uploads/"+filename1+".txt", "r")
            
         text1=key.readlines()
         text2=paper.readlines()
         str1=''.join(text1)
         str2=''.join(text2)
         sent_text1=str1.split('\n')
         sent_text2=str2.split('\n')
         final_list=[]
         for z in sent_text1:
             for y in sent_text2:
                 if z==y:
                     final_list.append(z)
         from difflib import SequenceMatcher
         with open("uploads/"+filename+".txt") as file1,open("uploads/"+filename1+".txt") as file2:
             file1_data=file1.read()
             file2_data=file2.read()
             Similarity=SequenceMatcher(None,file1_data,file2_data).ratio()
             A=Similarity*100
    with open('results.txt','a') as f2:
                         f2.write(filename1)
                         f2.write(":")
                         f2.write(str(A))
                         f2.write('\n')
                         f2.close()
    return render_template("predict.html",value=A)


@app.route('/generate')
def generate():
    file_path = "results.txt"

    @after_this_request
    def remove_file(response):    
        os.remove(file_path)
        return response
    
    return send_file("results.txt", mimetype='application/txt',attachment_filename='results.txt' ,as_attachment=True, cache_timeout=0)


                        
if(__name__=="__main__"):
    app.run(debug=True)