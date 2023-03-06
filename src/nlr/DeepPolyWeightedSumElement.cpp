/*********************                                                        */
/*! \file DeepPolyWeightedSumElement.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze Andrew Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "DeepPolyWeightedSumElement.h"
#include "FloatUtils.h"

#include <string.h>

namespace NLR {

DeepPolyWeightedSumElement::DeepPolyWeightedSumElement( Layer *layer )
    : _workLb( NULL )
    , _workUb( NULL )
{
    _layer = layer;
    _size = layer->getSize();
    _layerIndex = layer->getLayerIndex();
}

DeepPolyWeightedSumElement::~DeepPolyWeightedSumElement()
{
    freeMemoryIfNeeded();
}

void DeepPolyWeightedSumElement::execute
( const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore )
{
    log( "Executing..." );
    ASSERT( hasPredecessor() );
    allocateMemory();
    getConcreteBounds();
    // Compute bounds with back-substitution
    computeBoundWithBackSubstitution( deepPolyElementsBefore );
    log( "Executing - done" );
}

void DeepPolyWeightedSumElement::computeBoundWithBackSubstitution
( const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore )
{
    log( "Computing bounds with back substitution..." );

    // Start with the symbolic upper-/lower- bounds of this layer with
    // respect to its immediate predecessor.
    Map<unsigned, unsigned> predecessorIndices = getPredecessorIndices();

    unsigned counter = 0;
    unsigned numPredecessors = predecessorIndices.size();
    ASSERT( numPredecessors > 0 );
    // # The invariant we are maintaining:
    // thisLayer <= ( residualUb * residualLayer for each residualLayer ) +
    //                _work1SymbolicUb * currentElement + _workSymbolicUpperBias;
    // thisLayer >= ( residualLb * residualLayer for each residualLayer ) +
    //                _work1SymbolicLb * currentElement + _workSymbolicLowerBias;

    unsigned predecessorIndex = 0;
    for ( const auto &pair : predecessorIndices )
    {
        predecessorIndex = pair.first;
        if ( counter < numPredecessors - 1 )
        {
            log( Stringf( "Adding residual from layer %u...",
                          predecessorIndex ) );
            allocateMemoryForResidualsIfNeeded( predecessorIndex, pair.second );
            const double *weights = _layer->getWeights( predecessorIndex );
            memcpy( _residualLb[predecessorIndex], weights,
                    _size * pair.second * sizeof(double) );
            memcpy( _residualUb[predecessorIndex], weights,
                    _size * pair.second * sizeof(double) );
            ++counter;
            log( Stringf( "Adding residual from layer %u - done", pair.first ) );
        }
    }

    log( Stringf( "Computing symbolic bounds with respect to layer %u...",
                  predecessorIndex ) );
    DeepPolyElement *precedingElement =
        deepPolyElementsBefore[predecessorIndex];
    unsigned sourceLayerSize = precedingElement->getSize();

    const double *weights = _layer->getWeights( predecessorIndex );
    memcpy( _work1SymbolicLb,
            weights, _size * sourceLayerSize * sizeof(double) );
    memcpy( _work1SymbolicUb,
            weights, _size * sourceLayerSize * sizeof(double) );

    double *bias = _layer->getBiases();
    memcpy( _workSymbolicLowerBias, bias, _size * sizeof(double) );
    memcpy( _workSymbolicUpperBias, bias, _size * sizeof(double) );

    DeepPolyElement *currentElement = precedingElement;
    DeepPolyElement *preSizeElement = precedingElement;
    concretizeSymbolicBound( _work1SymbolicLb, _work1SymbolicUb,
                             _workSymbolicLowerBias,
                             _workSymbolicUpperBias,
                             currentElement, deepPolyElementsBefore );
    log( Stringf( "Computing symbolic bounds with respect to layer %u - done",
                  predecessorIndex ) );

    while ( currentElement->hasPredecessor() )
    {
        // We have the symbolic bounds in terms of the current abstract
        // element--currentElement, stored in _work1SymbolicLb,
        // _work1SymbolicUb, _workSymbolicLowerBias, _workSymbolicLowerBias,
        // now compute the symbolic bounds in terms of currentElement's
        // predecessor.
        predecessorIndices = currentElement->getPredecessorIndices();
        counter = 0;
        numPredecessors = predecessorIndices.size();
        ASSERT( numPredecessors > 0 );
        for ( const auto &pair : predecessorIndices )
        {
            predecessorIndex = pair.first;
            precedingElement = deepPolyElementsBefore[predecessorIndex];

            if ( counter < numPredecessors - 1 )
            {
                unsigned predecessorIndex = pair.first;
                log( Stringf( "Adding residual from layer %u...",
                              predecessorIndex ) );
                allocateMemoryForResidualsIfNeeded( predecessorIndex,
                                                    pair.second );
                // Do we need to add bias here?
                currentElement->symbolicBoundInTermsOfPredecessor
                    ( _work1SymbolicLb, _work1SymbolicUb, NULL, NULL,
                      _residualLb[predecessorIndex],
                      _residualUb[predecessorIndex],
                      _size, precedingElement );
                ++counter;
                log( Stringf( "Adding residual from layer %u - done", pair.first ) );
            }
        }

        std::fill_n( _work2SymbolicLb, _size * precedingElement->getSize(), 0 );
        std::fill_n( _work2SymbolicUb, _size * precedingElement->getSize(), 0 );
        currentElement->symbolicBoundInTermsOfPredecessor
            ( _work1SymbolicLb, _work1SymbolicUb, _workSymbolicLowerBias,
              _workSymbolicUpperBias, _work2SymbolicLb, _work2SymbolicUb,
              _size, precedingElement );

        // The symbolic lower-bound is
        // _work2SymbolicLb * precedingElement + residualLb1 * residualElement1 +
        // residualLb2 * residualElement2 + ...
        // If the precedingElement is a residual source layer, we can merge
        // in the residualWeights, and remove it from the residual source layers.
        if ( _residualLayerIndices.exists( predecessorIndex ) )
        {
            log( Stringf( "merge residual from layer %u...", predecessorIndex ) );
            // Add weights of this residual layer
            for ( unsigned i = 0; i < _size * precedingElement->getSize(); ++i )
            {
                _work2SymbolicLb[i] += _residualLb[predecessorIndex][i];
                _work2SymbolicUb[i] += _residualUb[predecessorIndex][i];
            }
            _residualLayerIndices.erase( predecessorIndex );
            std::fill_n( _residualLb[predecessorIndex],
                         _size * precedingElement->getSize(), 0 );
            std::fill_n( _residualUb[predecessorIndex],
                         _size * precedingElement->getSize(), 0 );
            log( Stringf( "merge residual from layer %u - done", predecessorIndex ) );
        }

        DEBUG({
                // Residual layers topologically after precedingElement should
                // have been merged already.
                for ( const auto &residualLayerIndex : _residualLayerIndices )
                {
                    ASSERT( residualLayerIndex < predecessorIndex );
                }
            });

        double *temp = _work1SymbolicLb;
        _work1SymbolicLb = _work2SymbolicLb;
        _work2SymbolicLb = temp;

        temp = _work1SymbolicUb;
        _work1SymbolicUb = _work2SymbolicUb;
        _work2SymbolicUb = temp;

        currentElement = precedingElement;
        concretizeSymbolicBound( _work1SymbolicLb, _work1SymbolicUb,
                                 _workSymbolicLowerBias, _workSymbolicUpperBias,
                                 currentElement, deepPolyElementsBefore );
    }
    ASSERT( _residualLayerIndices.empty());

    auto intra_depends = [&](unsigned i, unsigned j) {
        unsigned pre_size = preSizeElement->getSize();
        std::vector<double> w1(pre_size), w2(pre_size), wp(pre_size);
        for (unsigned k = 0; k < pre_size; ++ k) {
            w1[k] = _work1SymbolicLb[k * _size + i];
            w2[k] = _work1SymbolicLb[k * _size + j];
        }
        double b1 = 0, b2 = 0;
        if (_workSymbolicLowerBias) {
            b1 = _workSymbolicLowerBias[i];
            b2 = _workSymbolicUpperBias[j];
        }

        bool found = true;
        unsigned nonzero_index = 0;
        while (w1[nonzero_index] == 0 or w2[nonzero_index] == 0) {
            nonzero_index += 1;
            if (nonzero_index == pre_size) {
                found = false;
                break;
            }
        }

        double _min = 0.0, _max = 0.0, bp = 0.0;
//        String s1 = "lower: ", s2 = "upper: ", s3 = "wp: ";

        if (found) {
            double coefficient = w1[nonzero_index] / w2[nonzero_index];
            for (unsigned index = nonzero_index; index < pre_size; ++ index) {
                wp[index] = w1[index] - coefficient * w2[index];
            }
            bp = b1 - coefficient * b2;
            _min = _max = bp;
            for (unsigned k = 0; k < pre_size; ++ k) {
                double lb = preSizeElement->getLowerBound(k);
                double ub = preSizeElement->getUpperBound(k);
                if (wp[k] < 0) {
                    _min += wp[k] * ub;
                    _max += wp[k] * lb;
                } else if (wp[k] > 0) {
                    _min += wp[k] * lb;
                    _max += wp[k] * ub;
                }
//                s3 += Stringf("%.8f ", wp[k]);
//                s1 += Stringf("%.8f ", lb);
//                s2 += Stringf("%.8f ", ub);
            }
        }
//        log(String("--------------------------------------------------------------"));
//        log(s1);
//        log(s3);
//        log(s2);
//        log(Stringf("min: %.8f, max: %.8f, bp: %.8f, b1: %.8f, b2: %.8f,", _min, _max, bp, b1, b2));
        return std::pair<bool, std::pair<double, double>> {found, {_min, _max}};
    };

    for (unsigned i = 0; i < _size; ++ i) {
        for (unsigned j = i + 1; j < _size; ++ j) {
            auto res1 = intra_depends(i, j);
            auto res2 = intra_depends(j, i);
            if (res1.first and res2.first) {
                String s = "Intra dependency: ";
                double min1 = res1.second.first, max1 = res1.second.second;
                double min2 = res2.second.first, max2 = res2.second.second;
                bool flag = true;
                if (max1 < 0 and max2 < 0) {
                    s += Stringf("(%u, %u) active ==> (%u, %u) inactive",
                                 getLayerIndex(), i, getLayerIndex(), j);
                } else if (min1 > 0 and min2 > 0) {
                    s += Stringf("(%u, %u) inactive ==> (%u, %u) active",
                                 getLayerIndex(), i, getLayerIndex(), j);
                } else if (max1 < 0 and 0 < min2) {
                    s += Stringf("(%u, %u) active ==> (%u, %u) active",
                                 getLayerIndex(), i, getLayerIndex(), j);
                } else if (min1 > 0 and 0 > max2){
                    s += Stringf("(%u, %u) inactive ==> (%u, %u) inactive",
                                 getLayerIndex(), i, getLayerIndex(), j);
                } else {
                    s += "None";
                    flag = false;
                }
                if (flag) {
                    bool has_certain = true;
                    if ( _lb[i] > 0 or _ub[i] < 0 )
                        has_certain = false;
                    if ( _lb[j] > 0 or _ub[j] < 0 )
                        has_certain = false;
                    s += Stringf("\n(%u, %u) : [%f, %f]", getLayerIndex(), i, _lb[i], _ub[i]);
                    s += Stringf("\n(%u, %u) : [%f, %f]", getLayerIndex(), j, _lb[j], _ub[j]);
                    if (has_certain)
                        printf("%s\n", s.ascii());
                }
            }
        }
    }

    log( "Computing bounds with back substitution - done" );
}

void DeepPolyWeightedSumElement::concretizeSymbolicBound
( const double *symbolicLb, const double*symbolicUb, double const
  *symbolicLowerBias, const double *symbolicUpperBias, DeepPolyElement
  *sourceElement, const Map<unsigned, DeepPolyElement *>
  &deepPolyElementsBefore )
{
    log( "Concretizing bound..." );
    std::fill_n( _workLb, _size, 0 );
    std::fill_n( _workUb, _size, 0 );

    concretizeSymbolicBoundForSourceLayer( symbolicLb, symbolicUb,
                                           symbolicLowerBias, symbolicUpperBias,
                                           sourceElement );

    for ( const auto &residualLayerIndex : _residualLayerIndices )
    {
        ASSERT( residualLayerIndex < sourceElement->getLayerIndex() );
        DeepPolyElement *residualElement =
            deepPolyElementsBefore[residualLayerIndex];
        concretizeSymbolicBoundForSourceLayer( _residualLb[residualLayerIndex],
                                               _residualUb[residualLayerIndex],
                                               NULL,
                                               NULL,
                                               residualElement );
    }
    for ( unsigned i = 0; i <_size; ++i )
    {
        if ( _lb[i] < _workLb[i] )
            _lb[i] = _workLb[i];
        if ( _ub[i] > _workUb[i] )
            _ub[i] = _workUb[i];

        log( Stringf( "concretizeSymbolicBound: Neuron%u_%u working LB: %f, UB: %f", getLayerIndex(), i, _workLb[i], _workUb[i] ) );
        log( Stringf( "concretizeSymbolicBound: Neuron%u_%u LB: %f, UB: %f", getLayerIndex(), i, _lb[i], _ub[i] ) );
    }

    log( "Concretizing bound - done" );
}

void DeepPolyWeightedSumElement:: concretizeSymbolicBoundForSourceLayer
( const double *symbolicLb, const double*symbolicUb, const double
  *symbolicLowerBias, const double *symbolicUpperBias, DeepPolyElement
  *sourceElement )
{
    // Get concrete bounds
    for ( unsigned i = 0; i < sourceElement->getSize(); ++i )
    {
        double sourceLb = sourceElement->getLowerBoundFromLayer( i );
        double sourceUb = sourceElement->getUpperBoundFromLayer( i );

        log( Stringf( "Bounds of neuron%u_%u: [%f, %f]\n", sourceElement->
                      getLayerIndex(), i, sourceLb, sourceUb ) );

        for ( unsigned j = 0; j < _size; ++j )
        {
            // Compute lower bound
            double weight = symbolicLb[i * _size + j];
            if ( weight >= 0 )
            {
                _workLb[j] += ( weight * sourceLb );
            } else
            {
                _workLb[j] += ( weight * sourceUb );
            }

            // Compute upper bound
            weight = symbolicUb[i * _size + j];
            if ( weight >= 0 )
            {
                _workUb[j] += ( weight * sourceUb );
            } else
            {
                _workUb[j] += ( weight * sourceLb );
            }
        }
    }

    for ( unsigned i = 0; i < _size; ++i )
    {
        if ( symbolicLowerBias )
            _workLb[i] += symbolicLowerBias[i];
        if ( symbolicUpperBias )
            _workUb[i] += symbolicUpperBias[i];
    }
}


void DeepPolyWeightedSumElement::symbolicBoundInTermsOfPredecessor
( const double *symbolicLb, const double*symbolicUb, double
  *symbolicLowerBias, double *symbolicUpperBias, double
  *symbolicLbInTermsOfPredecessor, double *symbolicUbInTermsOfPredecessor,
  unsigned targetLayerSize, DeepPolyElement *predecessor )
{
    unsigned predecessorIndex = predecessor->getLayerIndex();
    log( Stringf( "Computing symbolic bounds with respect to layer %u...",
                  predecessorIndex ) );
    unsigned predecessorSize = predecessor->getSize();

    double *weights = _layer->getWeights( predecessorIndex );
    double *biases = _layer->getBiases();

    // newSymbolicLb = weights * symbolicLb
    // newSymbolicUb = weights * symbolicUb
    matrixMultiplication( weights, symbolicLb,
                          symbolicLbInTermsOfPredecessor, predecessorSize,
                          _size, targetLayerSize );
    matrixMultiplication( weights, symbolicUb,
                          symbolicUbInTermsOfPredecessor, predecessorSize,
                          _size, targetLayerSize );

    // symbolicLowerBias = biases * symbolicLb
    // symbolicUpperBias = biases * symbolicUb
    if  ( symbolicLowerBias )
        matrixMultiplication( biases, symbolicLb, symbolicLowerBias, 1,
                              _size, targetLayerSize );
    if  ( symbolicUpperBias )
        matrixMultiplication( biases, symbolicUb, symbolicUpperBias, 1,
                              _size, targetLayerSize );
    log( Stringf( "Computing symbolic bounds with respect to layer %u - done",
                  predecessorIndex ) );
}

void DeepPolyWeightedSumElement::allocateMemoryForResidualsIfNeeded
( unsigned residualLayerIndex, unsigned residualLayerSize )
{
    _residualLayerIndices.insert( residualLayerIndex );
    unsigned matrixSize = residualLayerSize * _size;
    if ( !_residualLb.exists( residualLayerIndex ) )
    {
        double *residualLb = new double[matrixSize];
        std::fill_n( residualLb, matrixSize, 0 );
        _residualLb[residualLayerIndex] = residualLb;
    }
    if ( !_residualUb.exists( residualLayerIndex ) )
    {
        double *residualUb = new double[residualLayerSize * _size];
        std::fill_n( residualUb, matrixSize, 0 );
        _residualUb[residualLayerIndex] = residualUb;
    }
}

void DeepPolyWeightedSumElement::allocateMemory()
{
    freeMemoryIfNeeded();

    DeepPolyElement::allocateMemory();

    _workLb = new double[_size];
    _workUb = new double[_size];

    std::fill_n( _workLb, _size, FloatUtils::negativeInfinity() );
    std::fill_n( _workUb, _size, FloatUtils::infinity() );
}

void DeepPolyWeightedSumElement::freeMemoryIfNeeded()
{
    DeepPolyElement::freeMemoryIfNeeded();
    if ( _workLb )
    {
        delete[] _workLb;
        _workLb = NULL;
    }
    if ( _workUb )
    {
        delete[] _workUb;
        _workUb = NULL;
    }
    for ( auto const &pair : _residualLb )
    {
        delete[] pair.second;
    }
    _residualLb.clear();
    for ( auto const &pair : _residualUb )
    {
        delete[] pair.second;
    }
    _residualUb.clear();
    _residualLayerIndices.clear();
}

void DeepPolyWeightedSumElement::log( const String &message )
{
    if ( GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING )
        printf( "DeepPolyWeightedSumElement: %s\n", message.ascii() );
}

} // namespace NLR
