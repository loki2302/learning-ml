const expect = require('chai').expect;
const ML = require('ml');
const TOLERANCE = 0.00001;

// rows are users and columns are movies
const votes = new ML.Matrix([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0]
]);

const userToUserSimilarities = ML.Math.DistanceMatrix(votes, ML.Math.Similarity.cosine);
for(var userA = 0; userA < votes.rows; ++userA) {
    for(var userB = 0; userB < votes.rows; ++userB) {
        const similarity = userToUserSimilarities[userA][userB];
        console.log(`Similarity of user ${userA} and user ${userB} is ${similarity.toFixed(2)}`);
    }
}

const movieToMovieSimilarities = ML.Math.DistanceMatrix(votes.transpose(), ML.Math.Similarity.cosine);
for(var movieA = 0; movieA < votes.columns; ++movieA) {
    for(var movieB = 0; movieB < votes.columns; ++movieB) {
        const similarity = movieToMovieSimilarities[movieA][movieB];
        console.log(`Similarity of movie ${movieA} and movie ${movieB} is ${similarity.toFixed(2)}`);
    }
}

function getUserToUserSimilarity(userA, userB) {
    return userToUserSimilarities[userA][userB];
}

function getMovieToMovieSimilarity(movieA, movieB) {
    return movieToMovieSimilarities[movieA][movieB];
}

describe('ML.Math.Similarity.cosine', () => {
    it('should give a value of 1 for similar values', () => {
        const similarity = ML.Math.Similarity.cosine([1, 0], [1, 0]);
        expect(similarity).to.be.closeTo(1, TOLERANCE);
    });

    it('should give a value of 0 for unsimilar values', () => {
        const similarity = ML.Math.Similarity.cosine([1, 0], [0, 1]);
        expect(similarity).to.be.closeTo(0, TOLERANCE);
    });
});

describe('My super smart recommender', () => {
    it('should state that users 0 and 1 are similar', () => {
        const similarity = getUserToUserSimilarity(0, 1);
        expect(similarity).to.be.closeTo(1, TOLERANCE);
    });

    it('should state that users 0 and 2 are unsimilar', () => {
        const similarity = getUserToUserSimilarity(0, 2);
        expect(similarity).to.be.closeTo(0, TOLERANCE);
    });

    it('should state that movies 0 and 1 are similar', () => {
        const similarity = getMovieToMovieSimilarity(0, 1);
        expect(similarity).to.be.closeTo(1, TOLERANCE);
    });

    it('should state that movies 0 and 2 are unsimilar', () => {
        const similarity = getMovieToMovieSimilarity(0, 2);
        expect(similarity).to.be.closeTo(0, TOLERANCE);
    });

    it('should state that similarity of movies 0 and 3 is undefined', () => {
        const similarity = getMovieToMovieSimilarity(0, 3);
        expect(similarity).to.be.NaN;
    });
});
