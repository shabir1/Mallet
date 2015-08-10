package com.classifier;

/**
 *@Author : Shabir Ahmad
 * @Date: 9-08-2015
 */

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Labeling;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;

/**\
 *
 */
public class Demo {

    public static void main(String args[]) throws IOException {
        //prepare instance transformation pipeline
        ArrayList<Pipe> pipes = new ArrayList<Pipe>();
        pipes.add(new Target2Label());
        pipes.add(new CharSequence2TokenSequence());
        pipes.add(new TokenSequence2FeatureSequence());
        pipes.add(new FeatureSequence2FeatureVector());
        SerialPipes pipe = new SerialPipes(pipes);

        //prepare training instances
        InstanceList trainingInstanceList = new InstanceList(pipe);
        trainingInstanceList.addThruPipe(new CsvIterator(new FileReader("resource/data.txt"),
                "(\\w+)\\s+(\\w+)\\s+(.*)",
                3, 2, 1));

        //prepare test instances
        InstanceList testingInstanceList = new InstanceList(pipe);
        testingInstanceList.addThruPipe(new CsvIterator(new FileReader("resource/test.txt"),
                "(\\w+)\\s+(\\w+)\\s+(.*)",
                3, 2, 1));
        ClassifierTrainer trainer = new NaiveBayesTrainer();
        Classifier classifier = trainer.train(trainingInstanceList);
        System.out.println("Accuracy: " + classifier.getAccuracy(testingInstanceList));
        System.out.println("test   "+classifier.getAverageRank(testingInstanceList));
        System.out.println("training  "+classifier.getAverageRank(trainingInstanceList));


    }

    }
