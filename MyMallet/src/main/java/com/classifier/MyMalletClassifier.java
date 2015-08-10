package com.classifier;

import cc.mallet.classify.Classification;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.pipe.*;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

import java.util.ArrayList;

/**
 * created by shabir 8-8-2015
 *
 */
public class MyMalletClassifier extends Classifier {

    public Classifier trainClassifier(InstanceList trainingInstance) {

        ClassifierTrainer trainer = new NaiveBayesTrainer();
        return trainer.train(trainingInstance);
    }

    public Pipe getInstancePipe() {
        ArrayList<Pipe> pipes = new ArrayList<Pipe>();
        pipes.add(new Target2Label());
        pipes.add(new CharSequence2TokenSequence());
        pipes.add(new TokenSequence2FeatureSequence());
        pipes.add(new FeatureSequence2FeatureVector());
        SerialPipes pipe = new SerialPipes(pipes);

        return pipe;
    }

    @Override
    public Classification classify(Instance instance) {
        return null;
    }


}
