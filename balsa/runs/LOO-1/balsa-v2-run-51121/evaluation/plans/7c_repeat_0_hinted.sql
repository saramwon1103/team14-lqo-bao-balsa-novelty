/*+ HashJoin(an n it pi ci lt ml t)
 HashJoin(n it pi ci lt ml t)
 HashJoin(it pi ci lt ml t)
 HashJoin(pi ci lt ml t)
 HashJoin(ci lt ml t)
 NestLoop(lt ml t)
 NestLoop(ml t)
 SeqScan(an)
 SeqScan(n)
 SeqScan(it)
 SeqScan(pi)
 SeqScan(ci)
 SeqScan(lt)
 IndexScan(ml)
 IndexScan(t)
 Leading((an (n (it (pi (ci (lt (ml t)))))))) */
SELECT MIN(n.name) AS cast_member_name,
       MIN(pi.info) AS cast_member_info
FROM aka_name AS an,
     cast_info AS ci,
     info_type AS it,
     link_type AS lt,
     movie_link AS ml,
     name AS n,
     person_info AS pi,
     title AS t
WHERE an.name IS NOT NULL
  AND (an.name LIKE '%a%'
       OR an.name LIKE 'A%')
  AND it.info ='mini biography'
  AND lt.link IN ('references',
                  'referenced in',
                  'features',
                  'featured in')
  AND n.name_pcode_cf BETWEEN 'A' AND 'F'
  AND (n.gender='m'
       OR (n.gender = 'f'
           AND n.name LIKE 'A%'))
  AND pi.note IS NOT NULL
  AND t.production_year BETWEEN 1980 AND 2010
  AND n.id = an.person_id
  AND n.id = pi.person_id
  AND ci.person_id = n.id
  AND t.id = ci.movie_id
  AND ml.linked_movie_id = t.id
  AND lt.id = ml.link_type_id
  AND it.id = pi.info_type_id
  AND pi.person_id = an.person_id
  AND pi.person_id = ci.person_id
  AND an.person_id = ci.person_id
  AND ci.movie_id = ml.linked_movie_id;

