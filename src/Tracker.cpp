#include "Tracker.h"
#include "lh_tracking.h"



Tracker* Tracker::createVSDTracker(){
	return new LHTracker();
}
