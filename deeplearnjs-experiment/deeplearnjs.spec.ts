import {ENV, FeedEntry, NDArray, Tensor} from "deeplearn";
import { Graph, Array1D, Array2D, Session, SGDOptimizer, InCPUMemoryShuffledInputProviderBuilder, CostReduction } from 'deeplearn';
import {expect} from 'chai';

class LinearModel {
    constructor(
        private inputTensor: Tensor,
        private outputTensor: Tensor,
        private session: Session)
    {}

    async predict(input: number): Promise<number> {
        const testInput = Array1D.new([input]);
        const testFeedEntries: FeedEntry[] = [
            { tensor: this.inputTensor, data: <NDArray>testInput }
        ];

        const testOutput = this.session.eval(this.outputTensor, testFeedEntries);
        return await testOutput.val(0);
    }

    static async train(samples: TrainingSample[], numberOfBatches = 100, batchSize = 3): Promise<LinearModel> {
        const g = new Graph();
        const inputTensor = g.placeholder('input', [1]);
        const labelTensor = g.placeholder('label', [1]);
        const multiplier = g.variable('multiplier', <NDArray>Array2D.randNormal([1, 1]));
        const outputTensor = g.matmul(multiplier, inputTensor);
        const costTensor = g.meanSquaredCost(outputTensor, labelTensor);

        const session = new Session(g, ENV.math);
        const optimizer = new SGDOptimizer(0.00001);

        const inputs: Array1D[] = samples.map(sample => Array1D.new([sample.input]));
        const labels: Array1D[] = samples.map(sample => Array1D.new([sample.output]));

        const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([inputs, labels]);
        const [inputProvider, labelProvider] = shuffledInputProviderBuilder.getInputProviders();

        const feedEntries: FeedEntry[] = [
            { tensor: inputTensor, data: inputProvider },
            { tensor: labelTensor, data: labelProvider }
        ];

        for (let i = 0; i < numberOfBatches; i++) {
            session.train(costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);
        }

        return new LinearModel(inputTensor, outputTensor, session);
    }
}

class TrainingSample {
    constructor(
        public input: number,
        public output: number)
    {}
}

describe('deeplearn.js', () => {
    it('should work', async () => {
        const model = await LinearModel.train([
            new TrainingSample(1, 10),
            new TrainingSample(10, 100),
            new TrainingSample(100, 1000)
        ], 1000);

        expect(await model.predict(0.5)).to.be.closeTo(5, 0.01);
        expect(await model.predict(5)).to.be.closeTo(50, 0.1);
        expect(await model.predict(50)).to.be.closeTo(500, 1.0);
        expect(await model.predict(500)).to.be.closeTo(5000, 10.0);
        expect(await model.predict(5000)).to.be.closeTo(50000, 100.0);
    });
});
