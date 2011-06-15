////////////////////////////////////////////////////////////////////////
//  gpRepel : An R packge for GPU computing
//  GNU GPL
//

#ifndef __gpRepel_H__
#define __gpRepel_H__

#ifdef __cplusplus
extern "C"
{
#endif 

#include "inc/gpRtypes.h"


////////////////////////////////////////////////////////////////////////
// getDevice
//
// @param device            Device Id of the GPU in use by the current thread
//


void getDevice(PInteger device);

void gpRfirst(
        const PNumeric points, 
        const PInteger pNum, 
        const PInteger pDim,
        PNumeric pout
        );

void gprpostmave(PNumeric pint, PInteger a, PInteger b, PInteger win, PNumeric pout);

void gprpremave(PNumeric pint, PInteger a, PInteger b, PInteger win, PNumeric pout);

void gprmoverage( const PNumeric, const PInteger, const PInteger, const PInteger, PNumeric );

void gprbasavoff(const PNumeric pint, const PInteger a, const PInteger b, const PInteger win1, const PInteger win2, PNumeric pout);

void gprbasoroff(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout);

void gprdiff(PNumeric pint, PInteger a, PInteger b, PInteger, PNumeric pout);

void gprdiffrev(PNumeric pint, PInteger a, PInteger b, PInteger, PNumeric pout);

void gprmovemax(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout);

void gprup(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric pout);

void gprdown(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric pout);

void gpravgall(PNumeric pint, PInteger a, PInteger b, PNumeric pout);

void gprsdall(PNumeric pint, PInteger a, PInteger b, PNumeric pout);

void gprpeakmask(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric win2, PNumeric up, PNumeric pout);

void gprpeak2mask(PNumeric pint, PInteger a, PInteger b, PNumeric win1, PNumeric win2, PNumeric win3, PNumeric pout);

void gprmeanmax(PNumeric pint, PInteger a, PInteger b, PInteger win1, PNumeric pout);

#ifdef __cplusplus
}
#endif 

#endif     // __gpRepel_H__







